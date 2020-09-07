import torch
import numpy as np
from numpy import long
from torch.optim import lr_scheduler
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
from Newunet import Insensee_3Dunet
from torch import optim
import MRIdataset
from torch.utils.data import DataLoader
from advanced_model import DeepSupervision_U_Net
from ResNetUNet import ResNetUNet
from advanced_model import CleanU_Net
from ValModel import val_model
from criterions import sigmoid_dice
from transform import imageaug
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 5e-4


def train_model(model, criterion, optimizer, dataload, num_epochs=400):
    # model.load_state_dict(torch.load('./3dunet_model_save/weights_199.pth'))
    MAX = 0
    index = 0
    for epoch in range(num_epochs):
        save_loss = []
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('learning_rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y,_,_ in dataload:  # x:(1,1,C,H,W) y:(1,C,H,W)
            x = torch.squeeze(x)  # x:(C,H,W)
            y = torch.squeeze(y)  # y:(C,H,W)
            loss = 0
            for z in range(x.shape[0]):
                img_2d = x[z, :, :]
                img_2d_reshape = np.reshape(img_2d, (1,img_2d.shape[0], img_2d.shape[1]))

                label_2d = y[z, :, :]
                label_2d_reshape = np.reshape(label_2d,(1, label_2d.shape[0], label_2d.shape[1]))

                img_lab_foraug = np.concatenate((img_2d_reshape, label_2d_reshape),axis=0)
                img_2d_aug, label_2d_aug = imageaug(img_lab_foraug)

                img_2d_aug = torch.from_numpy(img_2d_aug)
                label_2d_aug = torch.from_numpy(label_2d_aug)

                img_2d_add1 = torch.unsqueeze(img_2d_aug, 0)
                img_2d_fortrain = torch.unsqueeze(img_2d_add1, 0)  # (1,1,H,W)
                label_2d_fortrain = torch.unsqueeze(label_2d_aug, 0)  # (1,H,W)

                optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零

                inputs = img_2d_fortrain.float().to(device)
                labels = label_2d_fortrain.long().to(device)
                outputs = model(inputs)  # (1,4,H,W)

                # loss_2d = criterion(outputs,labels)
                outputs_softmax = F.softmax(outputs,dim=1)
                _,_,_,loss_2d = criterion(outputs_softmax, labels)
                # print(loss)
                loss_2d.backward()  # 梯度下降,计算出梯度
                optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
                loss += loss_2d.item()
                save_loss.append(loss_2d)
            epoch_loss += loss
            mean_slice_loss = loss/x.shape[0]
            print("%d/%d,mean_slice_loss:%0.6f" % (step, dataset_size // dataload.batch_size, mean_slice_loss))
            step += 1
        np.savetxt('./3dunet_model_save/loss_%d.txt' % epoch, save_loss)
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss))

        if (epoch + 1) % 10 == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.98

        if (epoch+1) % 50 == 0:
            Mean_metric = val_model(model)
            if Mean_metric > MAX:
                MAX = Mean_metric
                print('------------------------')
                print(Mean_metric)
                print('------------------------')
                Best_weight = model.state_dict()
                index = epoch
    torch.save(Best_weight, './3dunet_model_save/BEST_weights_%d.pth' % index)


    return model


def train():
    # model = DeepSupervision_U_Net(in_channels=1, out_channels=4).to(device)
    # model = ResNetUNet(4).to(device)
    model = CleanU_Net(1,4).to(device)
    batch_size = 1
    # 损失函数
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_DC = sigmoid_dice
    # 梯度下降
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.98)
    liver_dataset = MRIdataset.LiverDataset(MRIdataset.imagepath, MRIdataset.labelpath, MRIdataset.img_ids,
                                            MRIdataset.label_ids, False)
    dataloader = DataLoader(liver_dataset, batch_size=1, shuffle=True, num_workers=4)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_model(model, criterion_DC, optimizer, dataloader)


if __name__ == '__main__':
    train()
