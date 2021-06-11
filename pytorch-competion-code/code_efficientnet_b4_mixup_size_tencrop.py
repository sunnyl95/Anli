# -*- coding: utf-8 -*-
import os
from moxing.framework import file
import warnings
warnings.filterwarnings("ignore")
print('current folder: ', os.getcwd())
# 将要用的模块加载到EVS
OBS_Name = 'lifang'
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
# model_path = '/home/work/.torch/models'
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
file.copy_parallel(os.path.join('s3://' + OBS_Name, 'garbage_classify_new'), file_folder)

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

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=40)
# print(model)

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224,(0.25,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
        transforms.Resize(330),
        transforms.TenCrop(288),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])



class BaseDataset(Dataset):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, transform):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        self.images = np.array(img_paths)
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        # x = self.preprocess_img(x)
        x = Image.open(x)
        x = self.transform(x)
        y = self.labels[idx]
        y = np.float32(y)
        return x, y


def data_flow(train_data_dir, val_data_dir, batch_size):  # need modify
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
    validation_img_paths, validation_labels = get_data(val_data_dir)
    # train_img_paths, validation_img_paths, train_labels, validation_labels = \
    #     train_test_split(img_paths, labels, test_size=0.25, random_state=0)
    print('training samples: %d, validation samples: %d' %
          (len(train_img_paths), len(validation_img_paths)))

    train_set = BaseDataset(train_img_paths, train_labels, transform=train_transforms)
    validation_set = BaseDataset(validation_img_paths, validation_labels, transform=val_transforms)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=validation_set, batch_size=batch_size, num_workers=0, shuffle=False)

    return train_set, validation_set, train_loader, val_loader


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

        if epoch <= 30:
            lr = 0.008
        if 30 < epoch < 60:
            lr = 0.001
        else:
            lr = 0.0005

        adjust_learning_rate(optimizer, lr)

        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_img, batch_labels in dataloaders[phase]:
                batch_img = batch_img.cuda()   #.to(device)
                batch_labels = batch_labels.long().cuda()   #.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        alpha = 0.2
                        lam = np.random.beta(alpha, alpha)
                        index = torch.randperm(batch_img.size(0)).cuda()  #.to(device)
                        inputs = lam * batch_img + (1 - lam) * batch_img[index, :]
                        outputs = model(inputs)
                        targets_a, targets_b = batch_labels, batch_labels[index]
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                        preds = torch.argmax(outputs, dim=1)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        input = batch_img  # input is a 5d tensor, target is 2d'
                        target= batch_labels
                        # print(input.size())
                        # print(target)
                        bs, ncrops, c, h, w = input.size()
                        result = model(input.view(-1, c, h, w))  # fuse batch size and ncrops
                        result_avg = result.view(bs, ncrops, -1).mean(1)  #
                        preds = torch.argmax(result_avg, dim=1)
                        loss = criterion(result_avg, batch_labels)

                running_loss += loss.item() * batch_img.size(0)
                # torch.Tensor
                running_corrects += torch.sum(preds.float() == batch_labels.float())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc and epoch_acc > 0.9:
                best_acc = epoch_acc
                torch.save(model.module, './eff_b4_mixup_size_tencrop.pth')
                file.copy('./eff_b4_mixup_size_tencrop.pth', 's3://' + OBS_Name + f'/eff_b4_mixup_size_tencrop/eff_b4_mixup_size_tencrop_acc_{epoch_acc}.pth')
                # best_model_wt = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_loss < least_loss:
                least_loss = epoch_loss
                torch.save(model.module, './eff_b4_mixup_size_tencrop.pth')
                file.copy('./eff_b4_mixup_size_tencrop.pth', 's3://' + OBS_Name + f'/eff_b4_mixup_size_tencrop/eff_b4_mixup_size_tencrop_loss_{epoch_loss}.pth')
        end = time.time()

        print('Training complete in {:.0f}s'.format((end - start)))
        print('Best val Acc: {:4f}'.format(best_acc))

    return model


train_set, validation_set, train_loader, val_loader = data_flow(train_data_dir=file_folder+'/Ftrain', 
                                                                val_data_dir=file_folder+'/Ftest',
                                                                batch_size=95)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_set), 'val': len(validation_set)}
model = torch.nn.DataParallel(model).cuda()
# model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda() #.to(device)
model = train_model(model, criterion, optimizer, num_epochs=150)
