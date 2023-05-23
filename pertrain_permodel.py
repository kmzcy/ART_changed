import os
from sched import scheduler
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
import utils.cnn_model as cnn_model
import utils.dataset_CIFAR100 as dp
from torch.utils.data import DataLoader
from final_predict import final_predict
import numpy as np
import time
import h5py

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
max_iter = 1
epochs = 5
batch_size = 8
the_device = 'cuda:2'
model_name = 'vgg19' 
dataset_name = 'cifra100'

binary_length = 2048
device = torch.device(the_device if torch.cuda.is_available() else "cpu") 

path_train_dataset = "save_weights/train_data_" + str(dataset_name) + '_' + str(model_name) + "_2048_fintuned.mat"
path_test_dataset = "save_weights/test_data_" + str(dataset_name) + '_' + str(model_name) + "_2048_fintuned.mat"

class predict(nn.Module):
    def __init__(self, input_size, classification):
        super(predict, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(input_size, classification),
            )
    
    def forward(self, x):
        y = self.classifier(x)
        return y

def dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_train = dp.ITFL_DataSet(
        os.getcwd(), transformations, 'train'
        )

    dset_test = dp.ITFL_DataSet(
        os.getcwd(), transformations, 'test'
        )

    num_train, num_test = len(dset_train), len(dset_test)

    dsets = (dset_train, dset_test)
    nums = (num_train, num_test)

    return nums, dsets

def train_model(code_length, classification, train_loader):

    model = cnn_model.CNNNet(model_name, code_length)
    model.to(device)
    model.train()
    model.eval()

    predict_model = predict(code_length, classification)
    predict_model.to(device)
    predict_model.train()

    optimizer_model = torch.optim.SGD(model.parameters(),
                                      lr=0.0000001, 
                                      momentum=0.9)

    optimizer = torch.optim.SGD(predict_model.parameters(),
                                lr=0.0001, 
                                momentum=0.9)

    scheduler_model = optim.lr_scheduler.ExponentialLR(optimizer_model, gamma = 0.0000001, last_epoch=-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.0001, last_epoch=-1)
    loss_function = torch.nn.CrossEntropyLoss()

    for iter in range(max_iter):
        for epoch in range(epochs):
            training_loss = 0.0
            for i, (x, y, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                optimizer_model.zero_grad

                output = model(x)
                output = predict_model(output)

                loss = loss_function(output, y)
                print('epoch: {}, step: {}, Loss: {:.2f}, '.format(epoch, i, loss))
                
                loss.backward()

                optimizer.step()
                optimizer_model.step()

                training_loss += loss.data.item() * x.size(0)

            scheduler.step()
            scheduler_model.step()

            training_loss /= len(train_loader.dataset)
            print('Iter:{}, Epoch: {}, total training Loss: {:.2f}, '.format(iter, epoch, training_loss))


    return model, predict_model

def start_model():
    _, dsets = dataset()
    des_train, des_test = dsets

    train_data_loader = DataLoader(des_train, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(des_test, shuffle=True, batch_size=batch_size)

    model, predict_model = train_model(binary_length, des_train.classification ,train_data_loader)

    model.eval()
    predict_model.eval()

    train_image = []
    trian_labels = []

    for step, (x, y, _) in enumerate(train_data_loader):
        x = x.to(device)
        y = y.to(device)
        
        outputs = model(x)
        outputs = outputs.detach()
        # save data
        for i in range(len(outputs)):
            train_image.append(outputs[i])
            trian_labels.append(y[i])

    for i in range(len(train_image)):
        train_image[i] = train_image[i].detach().to('cpu').numpy()
        trian_labels[i] = trian_labels[i].detach().to('cpu').numpy()
    
    print("len(train_image): ", len(train_image))
    print("len(trian_labels): ", len(trian_labels))

    with h5py.File(path_train_dataset, 'w') as f:
        f.create_dataset('images',data = train_image[:])
        f.create_dataset('labels',data = trian_labels[:])
        f.close()

    test_images = []
    test_labels = []

    loss_function = torch.nn.CrossEntropyLoss()

    correct = 0.0
    test_loss = 0.0 

    # 被分为正类的正类
    TP = []
    # 被分为负类的正类
    FN = []
    # 被分为正类的负类
    FP = []
    for i in range(des_test.classification):
        TP.append(0)
        FN.append(0)
        FP.append(0)
    recall = []
    precision = []

    start = time.time()
    for step, (x, y, _) in enumerate(test_data_loader):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)

        outputs = outputs.detach()

        # save data
        for i in range(len(outputs)):
            test_images.append(outputs[i])
            test_labels.append(y[i])

        outputs = predict_model(outputs)
        
        loss = loss_function(outputs, y)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        
        print("preds: ", preds)
        print("y: ", y)
        
        for i in range(len(preds)):
            if preds[i] == y[i]:
                TP[y[i]] = TP[y[i]] + 1
            else:
                FN[y[i]] = FN[y[i]] + 1 # 分错的样本对于它本身的类来说是分为负类的正类
                FP[preds[i]] = FP[preds[i]] + 1 # 分错的样本对于它被分到的类来说是分为正类的负类

        correct += preds.eq(y).sum()
    
    finish = time.time()

    test_loss = test_loss / len(test_data_loader.dataset)
    test_acc = correct.float() / len(test_data_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss,
        test_acc,
        finish - start
    ))

    print("TP: ", TP)
    print("FN: ", FN)
    print("FP: ", FP)

    for i in range(des_test.classification):
        print("i: ", i, " TP[",i ,"]: ", TP[i], " FN[",i ,"]: ", FN[i], " FP[",i ,"]: ", FP[i])
        recall.append(TP[i]/(TP[i] + FN[i]))
        precision.append(TP[i]/(TP[i] + FP[i]))

    for i in range(des_test.classification):
        print("recall of class " + str(i) + " is: " + str(recall[i]))
        print("the all :", TP[i] + FN[i])
        print("precision of class " + str(i) + " is: " + str(precision[i]))

    for i in range(len(test_images)):
        test_images[i] = test_images[i].detach().to('cpu').numpy()
        test_labels[i] = test_labels[i].detach().to('cpu').numpy()

    print("len(train_image): ", len(test_images))
    print("len(trian_labels): ", len(test_labels))

    with h5py.File(path_test_dataset, 'w') as f:
        f.create_dataset('images',data = test_images[:])
        f.create_dataset('labels',data = test_labels[:])
        f.close()

if __name__ == "__main__":
    start_model()
    