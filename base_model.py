import os
import torch
import torch.optim as optim
from torchvision import transforms
import utils.cnn_model as cnn_model
import utils.dataset_CIFAR100 as dp
from torch.utils.data import DataLoader
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
max_iter = 1
epochs = 5
batch_size = 8
the_device = 'cuda:2'
model_name = 'vgg19'

device = torch.device(the_device if torch.cuda.is_available() else "cpu") 

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

def train_model(classification, train_loader):

    model = cnn_model.CNNNet(model_name, classification)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=0, last_epoch=-1)
    loss_function = torch.nn.CrossEntropyLoss()

    for iter in range(max_iter):
        for epoch in range(epochs):
            training_loss = 0.0
            correct = 0.0

            start = time.time()
            for i, (x, y, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output = model(x)
        
                loss = loss_function(output, y)
                print('epoch: {}, step: {}, Loss: {:.2f}, '.format(epoch, i, loss))
                
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * x.size(0)

                _, preds = output.max(1)
                correct += preds.eq(y).sum()

            finish = time.time()
            train_acc = correct.float() / len(train_loader.dataset)
            print('train set: Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                train_acc,
                finish - start
            ))

        scheduler.step()
        training_loss /= len(train_loader.dataset)
        print('Iter:{}, Epoch: {}, total training Loss: {:.2f}, '.format(iter, epoch, training_loss))

    return model


def start_model():
    _, dsets = dataset()
    des_train, des_test = dsets

    train_data_loader = DataLoader(des_train, shuffle=True, batch_size=batch_size)
    test_data_loader = DataLoader(des_test, shuffle=True, batch_size=batch_size)

    model = train_model(des_train.classification ,train_data_loader)
    # 如果不需要训练模型的话就用下面的
    # model = cnn_model.CNNNet(model_name, des_train.classification)

    model.eval()
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

    for i in range(des_test.classification):
        recall.append(TP[i]/(TP[i] + FN[i]))
        precision.append(TP[i]/(TP[i] + FP[i]))

    for i in range(des_test.classification):
        print("recall of class " + str(i) + " is: " + str(recall[i]))
        print("the all :", TP[i] + FN[i])
        print("precision of class " + str(i) + " is: " + str(precision[i]))

if __name__ == "__main__":
    start_model()
    