import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from tqdm import tqdm  # プログレスバーを表示しながら学習の進行状況を確認

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # Hugging Faceのトランスフォーマーモデルをインポート

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase（小文字に変換）
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# GloVeの事前学習済みエンベディングをロード
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# 質問文をGloVeエンベディングに変換する関数
def get_question_embedding(question, embeddings_index, embedding_dim=300):
    words = process_text(question).split()
    embeddings = []
    for word in words:
        embedding = embeddings_index.get(word)
        if embedding is not None:
            embeddings.append(embedding)
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, embeddings_index, transform=None, answer=True, embedding_dim=300):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.embeddings_index = embeddings_index
        self.embedding_dim = embedding_dim

        # question / answerの辞書を作成
        # self.question2idx = {}
        self.answer2idx = {}
        # self.idx2question = {}
        self.idx2answer = {}

        # 分散表現では事前学習されたエンベディングを使用して単語を直接ベクトル化するため、辞書の作成は不要
        # # 質問文に含まれる単語を辞書に追加
        # for question in self.df["question"]:
        #     question = process_text(question)
        #     words = question.split(" ")
        #     for word in words:
        #         if word not in self.question2idx:
        #             self.question2idx[word] = len(self.question2idx)
        # self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        # self.question2idx = dataset.question2idx  # 質問文の辞書を訓練データから更新する処理は不要
        self.answer2idx = dataset.answer2idx  # 回答の辞書を訓練データの辞書に更新するための処理は必要
        # self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        # one-hot表現に変換する処理は不要
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # question_words = self.df["question"][idx].split(" ")
        # for word in question_words:
        #     try:
        #         question[self.question2idx[word]] = 1  # one-hot表現に変換
        #     except KeyError:
        #         question[-1] = 1  # 未知語
        question_embedding = get_question_embedding(self.df["question"][idx], self.embeddings_index, self.embedding_dim)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            # return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)
            return image, torch.tensor(question_embedding, dtype=torch.float32), torch.tensor(answers), int(mode_answer_idx)

        else:
            # return image, torch.Tensor(question)
            return image, torch.tensor(question_embedding, dtype=torch.float32)

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    # def __init__(self, vocab_size: int, n_answer: int):
    def __init__(self, embedding_dim: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()
        # self.text_encoder = nn.Linear(vocab_size, 512)
        self.text_encoder = nn.Linear(embedding_dim, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    # def forward(self, image, question):
    def forward(self, image, question_embedding):
        image_feature = self.resnet(image)  # 画像の特徴量
        # question_feature = self.text_encoder(question)  # テキストの特徴量
        question_feature = self.text_encoder(question_embedding)

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


class VQAModelWithTransformers(nn.Module):
    def __init__(self, model_name="facebook/bart-large"):
        super().__init__()
        self.resnet = ResNet18()
        self.text_encoder = nn.Linear(300, 512)
        self.fc = nn.Linear(512 + 512, 512)
        self.answer_generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, image, question_embedding):
        image_feature = self.resnet(image)
        question_feature = self.text_encoder(question_embedding)
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x

    def generate_answer(self, question_embedding, max_length=50):
        inputs = self.tokenizer.encode(" ".join(map(str, question_embedding.tolist())), return_tensors="pt")
        outputs = self.answer_generator.generate(inputs, max_length=max_length, num_beams=5, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0
    batch_count = len(dataloader)

    start = time.time()
    # for image, question, answers, mode_answer in dataloader:
    for batch_idx, (image, question, answers, mode_answer) in enumerate(tqdm(dataloader, desc="Training")):
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / batch_count, total_acc / batch_count, simple_acc / batch_count, time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0
    batch_count = len(dataloader)

    start = time.time()
    # for image, question, answers, mode_answer in dataloader:
    for batch_idx, (image, question, answers, mode_answer) in enumerate(tqdm(dataloader, desc="Evaluating")):
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / batch_count, total_acc / batch_count, simple_acc / batch_count, time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # GloVeのエンベディングファイルのパスを指定
    glove_embeddings = load_glove_embeddings('/content/data/glove.6B.300d.txt')

    # dataloader / model
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # データ拡張を追加
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),   # ランダムなリサイズクロップ
        transforms.RandomHorizontalFlip(),   # 水平反転
        transforms.RandomRotation(10),       # ランダムな回転
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # カラージッター
        transforms.ToTensor(),               # テンソル変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),       # リサイズ
        transforms.ToTensor(),               # テンソル変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    train_dataset = VQADataset(df_path="/content/data/train.json", image_dir="/content/data/train", embeddings_index=glove_embeddings, transform=train_transform)
    test_dataset = VQADataset(df_path="/content/data/valid.json", image_dir="/content/data/valid", embeddings_index=glove_embeddings, transform=test_transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    # model = VQAModel(embedding_dim=300, n_answer=len(train_dataset.answer2idx)).to(device)
    model = VQAModelWithTransformers().to(device)

    # optimizer / criterion
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学習率スケジューリング

    start_epoch = 0
    num_epoch = 20
    # num_epoch = 1

    try:
        checkpoint = torch.load("checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded: Starting from epoch {start_epoch}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    # train model
    for epoch in range(start_epoch, num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_simple_acc, val_time = eval(model, test_loader, optimizer, criterion, device)  # 検証データで評価
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"{train_simple_acc:.4f}\n"
              f"val time: {val_time:.2f} [s]\n"
              f"val loss: {val_loss:.4f}\n"
              f"val acc: {val_acc:.4f}\n"
              f"val simple acc: {val_simple_acc:.4f}")
        
        scheduler.step()  # 学習率の更新

        # エポックごとにモデルの状態を保存
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, "checkpoint.pth")

        # 早期終了のチェック
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # ベストモデルの保存
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # 提出用ファイルの作成
        model.eval()
        submission = []
        # for image, question in test_loader:
        for image, question in tqdm(test_loader, desc="Creating Submission"):
            image, question = image.to(device), question.to(device)
            pred = model(image, question)
            pred = pred.argmax(1).cpu().item()
            submission.append(pred)

        submission = [train_dataset.idx2answer[id] for id in submission]
        submission = np.array(submission)
        # torch.save(model.state_dict(), "model.pth")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        # np.save("submission.npy", submission)
        np.save(f"submission_epoch_{epoch+1}.npy", submission)

if __name__ == "__main__":
    main()
