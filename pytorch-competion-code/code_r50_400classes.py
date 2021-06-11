# -*- coding: utf-8 -*-
import os
from moxing.framework import file
import warnings
warnings.filterwarnings("ignore")

# 将要用的模块加载到EVS
OBS_Name = 'jian1299'
dst_folder = './modules'
if not file.exists(dst_folder):
    file.mk_dir(dst_folder)

src_folder = 's3://' + 'jian1299' + '/modules'
src_name_2 = os.path.join(src_folder, 'torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')
dst_name_2 = os.path.join(dst_folder, 'torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')

file.copy(src_name_2, dst_name_2)
print(os.listdir('./modules'))

# 将预训练模型加载到EVS
model_path = '/home/work/.cache/torch/checkpoints'
if not file.exists(model_path):
    file.make_dirs(model_path)
src_model = os.path.join(src_folder, 'se_resnext101_32x4d-3b2fe3d8.pth')
dst_model = os.path.join(model_path, 'se_resnext101_32x4d-3b2fe3d8.pth')
file.copy(src_model, dst_model)
print(os.listdir('/home/work/.cache/torch/checkpoints'))

os.system('pip install '+ './modules/torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')
os.system('pip install torchvision==0.3.0')
# os.system('pip install '+ 'efficientnet_pytorch')

file_folder = '/cache/train_data/origin'
# file_folder_train = '/cache/train_data/train'
# file_folder_test = '/cache/train_data/test'
file_folder_class400 = '/cache/train_data/class400'
if not file.exists(file_folder):
    file.make_dirs(file_folder)
# if not file.exists(file_folder_train):
#     file.make_dirs(file_folder_train)
# if not file.exists(file_folder_test):
#     file.make_dirs(file_folder_test)
if not file.exists(file_folder_class400):
    file.make_dirs(file_folder_class400)
# file.copy_parallel(os.path.join('s3://' + 'jian1299', 'train'), file_folder_train)
# file.copy_parallel(os.path.join('s3://' + 'jian1299', 'test'), file_folder_test)
file.copy_parallel(os.path.join('s3://' + 'jian1299', 'origins'), file_folder)
file.copy_parallel(os.path.join('s3://' + 'jian1299', '400data'), file_folder_class400)

import random, codecs, math, time, copy
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
# from cnn_finetune import make_model
# import pretrainedmodels

model = make_model('se_resnext101_32x4d', num_classes=442, pretrained=True)

class MyModel(nn.Module):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        outputs0, outputs1, outputs2 = outputs.split([40, 399, 3], dim=1)
        outputs0 = F.softmax(torch.cat([outputs0, outputs2], dim=1))
        outputs0, outputs2 = outputs0.split([40, 3], dim=1)
        outputs1 = F.softmax(outputs1)
        outputs = torch.cat([outputs0, outputs1, outputs2], dim=1)
        return outputs

model = MyModel(base_model)

# ###### important parameters ##########
muti_gpu = True
epoch_num = 150
batch_size = 256
learning_rate = 0.01 * batch_size / 256
T = 10
# #####################################

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256, (0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
        transforms.Resize(330),
        transforms.CenterCrop(288),
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


def data_flow(train_data_dir_1, train_data_dir_2, batch_size):
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
    train_img_paths_1, train_labels_1 = get_data(train_data_dir_1)
    train_img_paths_2, train_labels_2 = get_data(train_data_dir_2)
    train_img_paths = train_img_paths_1 + train_img_paths_2
    train_labels = train_labels_1 + train_labels_2
    print('training samples: %d' %
          (len(train_img_paths)))

    train_set = BaseDataset(train_img_paths, train_labels, transform=train_transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    return train_set, train_loader


def train_model(model, criterion, criterion2,  optimizer, num_epochs=1):
    least_loss_50 = 1000
    least_loss_100 = 1000
    least_loss_150 = 1000
    least_loss_200 = 1000

    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(num_epochs):
        start = time.time()
        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        if epoch <= 40:
            lr = learning_rate
        else:
            lr = learning_rate / 10
        adjust_learning_rate(optimizer, lr)

        running_loss = 0.0
        running_corrects = 0

        for batch_img, batch_labels in train_loader:
            batch_img = batch_img.cuda()
            batch_labels = batch_labels.long().cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(batch_img.size(0)).cuda()
                inputs = lam * batch_img + (1 - lam) * batch_img[index, :]
                targets_a, targets_b = batch_labels, batch_labels[index]
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

            with torch.no_grad():
                output_T_1 = T_model_1(batch_img)
                output_T_2 = T_model_2(batch_img)
                output_T = F.softmax(output_T_1+output_T_2, dim=1)

            with torch.set_grad_enabled(True):
                outputs_S = F.log_softmax(outputs / T, dim=1)
                outputs_T = F.softmax(output_T / T, dim=1)
                # outputs_T = outputs_T.long().cuda()
                loss2 = criterion2(outputs_S, outputs_T) * T * T
                loss = loss * (1 - alpha) + loss2 * alpha
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * batch_img.size(0)
            running_corrects += torch.sum(preds.float() == batch_labels.float())

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch >= 15 and epoch < 40 and epoch_loss <= least_loss_50:
            least_loss_50 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se101_50.pth')
            else:
                torch.save(model, 'se101_50.pth')
             file.copy('./se101_50.pth', 's3://' + OBS_Name + './se101_400class/se101_50.pth')
        elif epoch >= 40 and epoch < 70 and epoch_loss <= least_loss_100:
            least_loss_100 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se101_120.pth')
            else:
                torch.save(model, 'se101_120.pth')
             file.copy('./se101_120.pth', 's3://' + OBS_Name + './se101_400class/se101_120.pth')
        elif epoch >= 70 and epoch < 100 and epoch_loss <= least_loss_150:
            least_loss_150 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se101_150.pth')
            else:
                torch.save(model, 'se101_150.pth')
            file.copy('./se101_150.pth', 's3://' + OBS_Name + './se101_400class/se101_150.pth')
        elif epoch >= 100 and epoch_loss <= least_loss_200:
            least_loss_200 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se101_200.pth')
            else:
                torch.save(model, 'se101_200.pth')
             file.copy('./se101_200.pth', 's3://' + OBS_Name + './se101_400class/se101_200.pth')

        end = time.time()

        print('Training complete in {:.2f} mins'.format((end - start)/60))
        print('Best 0-50 train Loss : {:4f}'.format(least_loss_20))
        print('Best 50-100 train Loss : {:4f}'.format(least_loss_40))
        print('Best 100-150train Loss : {:4f}'.format(least_loss_80))
        print('Best 150-200train Loss : {:4f}'.format(least_loss_120))
        print('Best 150-200train Loss : {:4f}'.format(least_loss_160))

    return model


train_set, train_loader = data_flow(train_data_dir_1=file_folder_train,
                                    train_data_dir_2=file_folder_class400,
                                    batch_size=batch_size)
dataset_size = len(train_set)

if muti_gpu:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()
criterion2 = nn.KLDivLoss().cuda()
model = train_model(model, criterion, criterion2, optimizer, num_epochs=epoch_num)