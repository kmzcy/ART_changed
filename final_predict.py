import time
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from os.path import join
import h5py
  
max_iter = 1
epochs = 10

class final_predict(nn.Module):
    def __init__(self, input_size, classification):
        super(final_predict, self).__init__()
        self.W1 = nn.Parameter(torch.randn(input_size, 4096, requires_grad=True))
        self.b1 = nn.Parameter(torch.zeros(4096, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(4096, 2048, requires_grad=True))
        self.b2 = nn.Parameter(torch.zeros(2048, requires_grad=True))
        self.W3 = nn.Parameter(torch.randn(2048, classification, requires_grad=True))
        self.b3 = nn.Parameter(torch.zeros(classification, requires_grad=True))

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def relu(self, X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
    
    def forward(self, x):
        x1 = torch.relu(x)
        x2 = torch.relu(x1@self.W1 + self.b1)
        x3 = torch.relu(x2@self.W2 + self.b2)
        x4 = x3@self.W3 + self.b3
        return x4

    def parameters(self):
        return self.params

class class_predict(nn.Module):
    def __init__(self, input_size, classification):
        super(final_predict, self).__init__()
        self.W = nn.Parameter(torch.randn(input_size, classification, requires_grad=True))
        self.b = nn.Parameter(torch.zeros(classification, requires_grad=True))

        self.params = [self.W, self.b]
    
    def forward(self, x):
        x1 = x@self.W1 + self.b1
        return x1

    def parameters(self):
        return self.params

def save_newData(ARTmodel, layer_num, train_loader, path_final_DataSet, usetype, device='cuda:0'):
    trainset = []
    labels = []
    template_labels = []
    correct = 0.0

    start = time.time()

    print("In the " + str(usetype) + " phase, create the dataset by concatenating the sample and the template. ")
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # 样本模板匹配，返回模板以及模板对应标签。
        template, temp_labels = ARTmodel.test_m(x, y, layer_num - 1) 
        print("traget_labels: ", temp_labels)
        print("y: ", y)

        # 计算模板的匹配准确率
        temp_labels = torch.tensor(temp_labels).to(device)
        correct += temp_labels.eq(y).sum()  

        # 模板样本拼接
        input = torch.cat((torch.Tensor([t.to('cpu').detach().numpy() for t in template]).to(device), x), dim = 1)
        input = input.detach()

        # 新样本，模板标签，样本真实标签存储
        for i in range(len(input)):
            trainset.append(input[i])
            template_labels.append(temp_labels[i]) # 匹配到模板的标签
            labels.append(y[i]) 

    finish = time.time()

    print("len(trainset): ", len(trainset))
    print("len(template_labels): ", len(template_labels))
    print("len(labels): ", len(labels))

    template_acc = correct.float() / len(train_loader.dataset)

    print('template maching acc: Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        template_acc,
        finish - start
    ))

    # 转化
    for i in range(len(trainset)):
        trainset[i] = trainset[i].to('cpu').numpy()
        template_labels[i] = template_labels[i].to('cpu').numpy()
        labels[i] = labels[i].to('cpu').numpy()

    # 存储    
    with h5py.File(path_final_DataSet, 'w') as f:
        f.create_dataset('images', data = trainset[:])
        f.create_dataset('template_labels', data = template_labels[:])
        f.create_dataset('labels', data = labels[:])
        f.close()
    print("save finished: ", path_final_DataSet)
    
class final_DataSet(Dataset):
    def __init__(self, path_final_DataSet, template_num = 1):

        all = h5py.File(path_final_DataSet, 'r')
       
        trainset_all = all["images"]
        template_labels_all = all["template_labels"]
        labels_all = all["labels"]

        self.trainset = []
        self.labels = []
        self.template_nums = []
        
        # 筛选属于特定模板的样本返回
        for i in range(len(trainset_all)):
            if template_labels_all[i] == template_num:
                self.trainset.append(trainset_all[i])
                self.template_nums.append(template_labels_all[i])
                self.labels.append(labels_all[i])

        self.count = len(self.trainset)              
        print(all.keys())
        print(len(self.trainset))
        print(len(self.labels))
        print(len(self.template_nums))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.trainset[idx], self.template_nums[idx], self.labels[idx]


def train_final(ARTmodel, layer_num, inputsize, classification, train_loader, dataset_name='ai2d', usetype='train', device='cuda:0'):

    batch_size = 8

    datasets = {}

    path_final_train_DataSet = "save_weights/final_data_" + str(dataset_name) + '_' + str(usetype) + ".mat"
    save_newData(ARTmodel, layer_num, train_loader, path_final_train_DataSet, usetype, device)

    # 把整个训练集根据模板的伪标签分为不同的训练集
    print("classification: ", classification)
    for i in range(classification + 1):
        datasets[i] = final_DataSet(path_final_train_DataSet, i - 1)

    data_loaders = {}
    for key in datasets:
        if datasets[key].count != 0:
            data_loaders[key] = DataLoader(datasets[key], 
                                        shuffle=True, 
                                        batch_size=batch_size)

    # 根据类别的个数classification，初始化对应类别个数的分类器
    classification_models = {}
    for i in range(classification + 1):
        classification_models[i] = final_predict(inputsize, classification)
    
    print("len(classification_models): ", len(classification_models))

    # class_predict = final_predict(2048, classification)
    # class_predict.to(device)

    optimizer = {}
    schedulers = {}
    loss_function = {}

    for i in range(classification + 1):
        optimizer[i] = torch.optim.Adam(classification_models[i].parameters(),
                                lr=0.0001,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0,
                                amsgrad=False)


        schedulers[i] = optim.lr_scheduler.CosineAnnealingLR(optimizer[i], 40, eta_min=0, last_epoch=-1)
        loss_function[i] = torch.nn.CrossEntropyLoss()

    # classification
    
    for key in data_loaders:
        classification_models[key].to(device)

        for iter in range(max_iter):
            for epoch in range(epochs):
                correct = 0.0
                training_loss = 0.0
                print("data_loaders[key]`s key:", key)
                for i, (x, _, y) in enumerate(data_loaders[key]):
                    x = x.to(device)
                    y = y.to(device)
                    optimizer[key].zero_grad()
                    output = classification_models[key](x)

                    loss = loss_function[key](output, y)
                    loss.backward()
                    optimizer[key].step()
                    training_loss += loss.data.item() * x.size(0)

                    _, preds = output.max(1)
                    correct += preds.eq(y).sum()
                    print("preds: ", preds)
                    print("y: ", y)
                    print('Class:{}, epoch: {}, step: {}, Loss: {:.2f}, '.format(key, epoch, i, loss))

                print("************************************")

                schedulers[key].step() 
                training_loss /= len(data_loaders[key].dataset)
                print("correct.float(): ", correct.float())
                print("len(data_loaders[key].dataset): ", len(data_loaders[key].dataset))
                train_acc = correct.float() / len(data_loaders[key].dataset)
                print('Class:{}, Iter:{}, Epoch: {}, train_acc: {:.2f}, total training Loss: {:.2f}, '.format(key, iter, epoch, train_acc, training_loss))

                print("-----------------------------------------------------")

        classification_models[key].to('cpu')
        
    # return classification_models, class_predict
    return classification_models
