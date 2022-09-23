import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import albumentations as albu
from torchinfo import summary


import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm  #コマンドラインで実行するとき
# from tqdm.notebook import tqdm  # jupyter で実行するとき

from utils import *

print(torch.__version__)
# リソースの指定（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 乱数シード固定（再現性の担保）
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
fix_seed(seed)

# データローダーのサブプロセスの乱数seedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# 分類先クラスの名称リスト
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 分類先クラス数　今回は10になる
n_output = len(list(set(classes)))

# 結果確認
print(n_output)

# Transformsの定義

# Transformsの定義

# 学習データ用: 正規化に追加で反転とRandomErasingを実施
transform_train = transforms.Compose([
  transforms.Resize(112),
  transforms.RandomHorizontalFlip(p=0.5), 
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5), 
  transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# 検証データ用 : 正規化のみ実施
transform = transforms.Compose([
  transforms.Resize(112),
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5)
])

albu_transforms = albu.Compose([
  albu.RandomRotate90(p=0.5),
  albu.RandomGamma(gamma_limit=(85, 115), p=0.2),
])

def albumentations_transform(image, transform=albu_transforms):    
  if transform:
    image_np = np.array(image)
    augmented = transform(image=image_np)
    image = Image.fromarray(augmented['image'])
  return image

data_transform = transforms.Compose([
  transforms.Lambda(albumentations_transform),
])
# データ取得用関数 Dataset

data_root = './data'

train_set = datasets.CIFAR10(
    root = data_root, train = True, 
    download = True, transform = transform_train)

# 検証データの取得
test_set = datasets.CIFAR10(
    root = data_root, train = False, 
    download = True, transform = transform)

# ミニバッチのサイズ指定
batch_size = 100

# 訓練用データローダー
# 訓練用なので、シャッフルをかける
train_loader = DataLoader(train_set, 
    batch_size = batch_size, shuffle = True)

# 検証用データローダー
# 検証時にシャッフルは不要
test_loader = DataLoader(test_set,  
    batch_size = batch_size, shuffle = False) 

net = models.resnet50(pretrained=True)
print(net)
net = net.to(device)
# summary(net, (100, 3, 224, 224))
print(net.fc)

#乱数の初期化
torch_seed()

#最終レイヤー関数の入力次元数を確認
fc_in_features = net.fc.in_features

# 分類先クラス名の定義

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 分類先クラス数　今回は10になる
n_output = len(classes)

#最終レイヤー関数の付替え
net.fc = nn.Linear(fc_in_features, n_output)

net = net.to(device)

lr = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# historyファイル初期化する
history = np.zeros((0, 5))


# 学習
num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)