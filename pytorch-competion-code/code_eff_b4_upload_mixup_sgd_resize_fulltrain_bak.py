# -*- coding: utf-8 -*-
import os
from moxing.framework import file
import warnings
warnings.filterwarnings("ignore")
print('current folder: ', os.getcwd())
# 将药用的模块加载到EVS
OBS_Name = 'rad5730-obs'
dst_folder = './modules'
if not file.exists(dst_folder):
    file.mk_dir(dst_folder)

src_folder = 's3://' + OBS_Name + '/modules'
# src_name_1 = os.path.join(src_folder, 'lookahead.py')
# dst_name_1 = os.path.join(dst_folder, 'lookahead.py')
src_name_2 = os.path.join(src_folder, 'torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')
dst_name_2 = os.path.join(dst_folder, 'torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')

# file.copy(src_name_1, dst_name_1)
file.copy(src_name_2, dst_name_2)
print(os.listdir('./modules'))

# 将预训练模型加载到EVS
model_path = '/home/work/.cache/torch/checkpoints'
if not file.exists(model_path):
    file.make_dirs(model_path)
src_model = os.path.join(src_folder, 'efficientnet-b4-6ed6700e.pth')
dst_model = os.path.join(model_path, 'efficientnet-b4-6ed6700e.pth')
file.copy(src_model, dst_model)
print(os.listdir('/home/work/.cache/torch/checkpoints'))

os.system('pip install '+ './modules/torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')
os.system('pip install '+ 'torchvision==0.3.0')
os.system('pip install '+ 'efficientnet_pytorch')

file_folder = '/cache/train_data'
if not file.exists(file_folder):
    file.make_dirs(file_folder)
file.copy_parallel(os.path.join('s3://' + OBS_Name, 'garbage_classify'), file_folder)

import random, codecs, math, time, copy
import numpy as np
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

has_trained_model_src = 's3://' + OBS_Name + '/eff_b4_full_train/eff_b4_full_train.pth'
file.copy(has_trained_model_src, './eff_b4_full_train.pth')
model = torch.load('./eff_b4_full_train.pth', map_location=torch.device('cpu'))
# model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=40)


train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class BaseDataset(Dataset):
    def __init__(self, img_paths, labels, transform):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        self.images = np.array(img_paths)
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = Image.open(x)
        x = self.transform(x)
        y = self.labels[idx]
        y = np.float32(y)
        return x, y


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]


def data_flow(train_data_dir, batch_size):
    def get_data(data_dir):
        label_files = glob(os.path.join(data_dir, '*.txt'))
        random.shuffle(label_files)
        img_paths = []
        labels = []
        for index, file_path in enumerate(label_files):
            with codecs.open(file_path, 'r', 'utf-8') as f:
                line = f.readline()
            line_split = line.strip().split(', ')
            if len(line_split) != 2:
                print('%s contain error lable' % os.path.basename(file_path))
                continue
            img_name = line_split[0]
            label = int(line_split[1])
            img_paths.append(os.path.join(data_dir, img_name))
            labels.append(label)
        return img_paths, labels
    # labels = to_categorical(labels, num_classes)
    train_img_paths, train_labels = get_data(train_data_dir)

    print('training samples: ',len(train_img_paths))

    train_set = BaseDataset(train_img_paths, train_labels, transform=train_transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=20, shuffle=True)

    return train_set, train_loader


def train_model(model, criterion, optimizer, num_epochs=1):
    best_acc = 0.0
    least_loss = 1000

    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(num_epochs):
        start = time.time()
        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        if epoch <= 40:
            lr = 0.004
        elif 40 < epoch <= 70:
            lr = 0.008
        else:
            lr = 0.0008
        adjust_learning_rate(optimizer, lr)

        running_loss = 0.0
        running_corrects = 0

        for batch_img, batch_labels in train_loader:
            batch_img = batch_img.cuda()   #to(device)
            batch_labels = batch_labels.long().cuda()   #.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(batch_img.size(0)).cuda() #.to(device)
                inputs = lam * batch_img + (1 - lam) * batch_img[index, :]
                targets_a, targets_b = batch_labels, batch_labels[index]
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * batch_img.size(0)
            running_corrects += torch.sum(preds.float() == batch_labels.float())

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_loss <= least_loss:
            least_loss = epoch_loss
            torch.save(model.module, './eff_b4_full_train_2.pth')
            file.copy('./eff_b4_full_train_2.pth', 's3://' + OBS_Name + '/eff_b4_full_train/eff_b4_full_train_2.pth')

        end = time.time()

        print('Training complete in {:.0f}s'.format((end - start)))
        print('Best train Loss: {:4f}\n'.format(least_loss))

    return model

train_set, train_loader = data_flow(train_data_dir=file_folder+'/train_data', batch_size=100)
dataset_size = len(train_set)
# model = model.to(device)
model = torch.nn.DataParallel(model).cuda()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()
model = train_model(model, criterion, optimizer, num_epochs=200)
