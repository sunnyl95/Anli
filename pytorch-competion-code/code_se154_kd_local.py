# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")
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
T = 10
alpha = 0.95
# ###################################

T_model_1 = torch.load('', map_location=torch.device('cpu'))
T_model_2 = torch.load('', map_location=torch.device('cpu'))

if muti_gpu:
    T_model_1 = torch.nn.DataParallel(T_model_1).cuda()
    T_model_2 = torch.nn.DataParallel(T_model_2).cuda()
else:
    T_model_1 = T_model_1.cuda()
    T_model_2 = T_model_2.cuda()

T_model_1.eval()
T_model_2.eval()

model = make_model('senet154', num_classes=43, pretrained=True)

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.16, 1)),
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

            with torch.no_grad():
                output_T_1 = T_model_1(batch_img)
                output_T_2 = T_model_2(batch_img)
            output_T_ori = output_T_1 + output_T_2
            output_T = output_T_ori - torch.mean(output_T_ori, dim=1,keepdim=True)
            output_T /= torch.std(output_T_ori, dim=1,keepdim=True)
            # output_T= torch.from_numpy(output_T)

            with torch.set_grad_enabled(True):
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(batch_img.size(0)).cuda()
                inputs = lam * batch_img + (1 - lam) * batch_img[index, :]
                targets_a, targets_b = batch_labels, batch_labels[index]
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

                output_S = F.log_softmax(outputs / T, dim=1)
                output_T = F.softmax(output_T / T, dim=1)
                output_T = output_T.cuda()

                loss2 = criterion2(output_S, output_T) * T * T

                loss = loss * (1 - alpha) + loss2 * alpha

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
                torch.save(model.module, 'se154_20.pth')
            else:
                torch.save(model, 'se154_20.pth')
        elif epoch >= 20 and epoch < 40 and epoch_loss <= least_loss_40:
            least_loss_40 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se154_40.pth')
            else:
                torch.save(model, 'se154_40.pth')
        elif epoch >= 40 and epoch < 80 and epoch_loss <= least_loss_80:
            least_loss_80 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se154_80.pth')
            else:
                torch.save(model, 'se154_80.pth')
        elif epoch >= 80 and epoch < 120 and epoch_loss <= least_loss_120:
            least_loss_120 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se154_120.pth')
            else:
                torch.save(model, 'se154_120.pth')
        elif epoch >= 120 and epoch_loss <= least_loss_160:
            least_loss_160 = epoch_loss
            if muti_gpu:
                torch.save(model.module, 'se154_160.pth')
            else:
                torch.save(model, 'se154_160.pth')

        end = time.time()

        print('Training complete in {:.2f} mins'.format((end - start)/60))
        print('Best 0-50 train Loss : {:4f}'.format(least_loss_20))
        print('Best 50-100 train Loss : {:4f}'.format(least_loss_40))
        print('Best 100-150train Loss : {:4f}'.format(least_loss_80))
        print('Best 150-200train Loss : {:4f}'.format(least_loss_120))
        print('Best 150-200train Loss : {:4f}'.format(least_loss_160))

    return model


train_set, train_loader = data_flow(train_data_dir='/disk/data/atrain', batch_size=batch_size)
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
