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
src_name_1 = os.path.join(src_folder, 'torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')
dst_name_1 = os.path.join(dst_folder, 'torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl')

src_name_2 = os.path.join(src_folder, 'lxj_ose154_120.pth')
dst_name_2 = os.path.join(dst_folder, 'lxj_ose154_120.pth')

src_name_3 = os.path.join(src_folder, 'senext101_200.pth')
dst_name_3 = os.path.join(dst_folder, 'senext101_200.pth')

src_name_4 = os.path.join(src_folder, 'dpn107_200.pth')
dst_name_4 = os.path.join(dst_folder, 'dpn107_200.pth')

file.copy(src_name_1, dst_name_1)
file.copy(src_name_2, dst_name_2)
file.copy(src_name_3, dst_name_3)
file.copy(src_name_4, dst_name_4)

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
os.system('pip install cnn_finetune')

file_folder_train = '/cache/train_data'
# file_folder_test = '/cache/train_data/test'
if not file.exists(file_folder_train):
    file.make_dirs(file_folder_train)
# if not file.exists(file_folder_test):
#     file.make_dirs(file_folder_test)
file.copy_parallel(os.path.join('s3://jian1299/origin'), file_folder_train)
# file.copy_parallel(os.path.join('s3://jian1299/origin'), file_folder_test)

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
from cnn_finetune import make_model

# ###### important parameters ##########
muti_gpu = True
epoch_num = 200
batch_size = 150
learning_rate = 0.01 * batch_size / 256
kd_alpha = 0.95
T = 10
# ###################################

T_model_1 = torch.load(dst_name_2, map_location=torch.device('cpu'))
T_model_2 = torch.load(dst_name_3, map_location=torch.device('cpu'))
T_model_3 = torch.load(dst_name_4, map_location=torch.device('cpu'))

if muti_gpu:
    T_model_1 = torch.nn.DataParallel(T_model_1).cuda()
    T_model_2 = torch.nn.DataParallel(T_model_2).cuda()
    T_model_3 = torch.nn.DataParallel(T_model_3).cuda()
else:
    T_model_1 = T_model_1.cuda()
    T_model_2 = T_model_2.cuda()
    T_model_3 = T_model_3.cuda()

T_model_1.eval()
T_model_2.eval()
T_model_3.eval()

model = make_model('se_resnext101_32x4d', num_classes=43, pretrained=True)


train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.16, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# test_transforms = transforms.Compose([
#         transforms.Resize(330),
#         transforms.FiveCrop(288),
#         transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#         transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
#     ])


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
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=2, shuffle=True)

    return train_set, train_loader


def train_model(model, criterion, criterion2,  optimizer, num_epochs=1):
    least_loss_20 = 1000
    least_loss_40 = 1000
    least_loss_80 = 1000
    least_loss_120 = 1000
    least_loss_160 = 1000

    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # def predict(model, img, c, h, w, bs, ncrops):
    #     with torch.no_grad():
    #         pred_score_several_crop = model(img.view(-1, c, h, w))
    #         pred_score_several_crop = pred_score_several_crop.cpu()
    #         pred_score_avg = pred_score_several_crop.view(bs, ncrops, -1).mean(1)
    #         pred_score = pred_score_avg.detach().numpy()
    #     return pred_score

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

            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(batch_img.size(0)).cuda()
            inputs = lam * batch_img + (1 - lam) * batch_img[index, :]
            
            with torch.no_grad():
                output_T_1 = T_model_1(inputs)
                output_T_2 = T_model_2(inputs)
                output_T_3 = T_model_3(inputs)
                output_T = output_T_1 + output_T_2 + output_T_3
                # output_T = output_T_ori - np.min(output_T_ori, axis=1, keepdims=True)
                # output_T /= (np.max(output_T_ori, axis=1, keepdims=True) - np.min(output_T_ori, axis=1, keepdims=True))
                # output_T= torch.from_numpy(output_T)

            with torch.set_grad_enabled(True):
                targets_a, targets_b = batch_labels, batch_labels[index]
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss_1 = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

                output_S = F.log_softmax(outputs / T, dim=1)
                output_T = F.softmax(output_T / T, dim=1)
                output_T = output_T.cuda()

                loss2 = criterion2(output_S, output_T) * T * T

                loss = loss_1 * (1 - kd_alpha) + loss2 * kd_alpha

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * batch_img.size(0)
            running_corrects += torch.sum(preds.float() == batch_labels.float())

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch < 20 and epoch_loss <= least_loss_20:
            least_loss_20 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'senext101_20.pth')
            else:
                torch.save(model, 'senext101_20.pth')
            file.copy('senext101_20.pth', 's3://jian1299/' + 'se101_3_kd/' + 'senext101_20.pth')
        elif epoch >= 20 and epoch < 40 and epoch_loss <= least_loss_40:
            least_loss_40 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'senext101_40.pth')
            else:
                torch.save(model, 'senext101_40.pth')
            file.copy('senext101_40.pth', 's3://jian1299/' + 'se101_3_kd/' + 'senext101_40.pth')
        elif epoch >= 40 and epoch < 80 and epoch_loss <= least_loss_80:
            least_loss_80 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'senext101_80.pth')
            else:
                torch.save(model, 'senext101_80.pth')
            file.copy('senext101_80.pth', 's3://jian1299/' + 'se101_3_kd/' + 'senext101_80.pth')
        elif epoch >= 80 and epoch < 120 and epoch_loss <= least_loss_120:
            least_loss_120 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'senext101_120.pth')
            else:
                torch.save(model, 'senext101_120.pth')
            file.copy('senext101_120.pth', 's3://jian1299/' + 'se101_3_kd/' + 'senext101_120.pth')
        elif epoch >= 120 and epoch_loss <= least_loss_160:
            least_loss_160 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'senext101_160.pth')
            else:
                torch.save(model, 'senext101_160.pth')
            file.copy('senext101_160.pth', 's3://jian1299/' + 'se101_kd/' + 'senext101_160.pth')

        end = time.time()

        print('Training complete in {:.2f} mins'.format((end - start)/60))
        print('Best 0-50 train Loss : {:4f}'.format(least_loss_20))
        print('Best 50-100 train Loss : {:4f}'.format(least_loss_40))
        print('Best 100-150train Loss : {:4f}'.format(least_loss_80))
        print('Best 150-200train Loss : {:4f}'.format(least_loss_120))
        print('Best 150-200train Loss : {:4f}'.format(least_loss_160))

    return model


train_set, train_loader = data_flow(train_data_dir=file_folder_train, batch_size=batch_size)
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
