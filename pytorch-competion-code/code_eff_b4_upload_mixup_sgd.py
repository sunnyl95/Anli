# -*- coding: utf-8 -*-
import os
from moxing.framework import file
import warnings
warnings.filterwarnings("ignore")
print('current folder: ', os.getcwd())
# 将药用的模块加载到EVS
OBS_Name = 'huaweinb'
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
os.system('pip install efficientnet_pytorch')

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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=40)
# print(resnet50)


train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        x = Image.open(x)
        x = self.transform(x)
        y = self.labels[idx]
        y = np.float32(y)
        return x, y


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]


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

    print('training samples: %d, validation samples: %d' %
          (len(train_img_paths), len(validation_img_paths)))

    train_set = BaseDataset(train_img_paths, train_labels, transform=train_transforms)
    validation_set = BaseDataset(validation_img_paths, validation_labels, transform=val_transforms)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=20, shuffle=True)
    val_loader = DataLoader(dataset=validation_set, batch_size=batch_size, num_workers=20, shuffle=False)

    return train_set, validation_set, train_loader, val_loader


def train_model(model, criterion, optimizer, num_epochs=1):
    best_acc = 0.0

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
            lr = 0.004
        else:
            lr = 0.0008
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
                batch_img = batch_img.cuda()   #to(device)
                batch_labels = batch_labels.long().cuda()   #.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(batch_img)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, batch_labels)

                    if phase == 'train':
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
                # torch.Tensor
                running_corrects += torch.sum(preds.float() == batch_labels.float())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.module, './eff_b4_mixup_sgd.pth')
                file.copy('./eff_b4_mixup_sgd.pth', 's3://' + OBS_Name + '/eff_b4_mixup_sgd/eff_b4_mixup_sgd.pth')
                # best_model_wt = copy.deepcopy(model.state_dict())

        end = time.time()

        print('Training complete in {:.0f}s'.format((end - start)))
        print('Best val Acc: {:4f}\n'.format(best_acc))

    return model

train_set, validation_set, train_loader, val_loader = data_flow(train_data_dir=file_folder+'/Ftrain', 
                                                                val_data_dir=file_folder+'/Ftest',
                                                                batch_size=100)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_set), 'val': len(validation_set)}
# model = model.to(device)
model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()  #.to(device)
model = train_model(model, criterion, optimizer, num_epochs=150)
