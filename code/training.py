import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle

from config import DGSDTAModelConfig
from model import DGSDTAModel
from dataset import DGSDTADataset
from utils import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        graph,seq_embed,seq_mask = data
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)

        optimizer.zero_grad()
        output = model(graph,seq_embed)
        loss = loss_fn(output, graph.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(graph.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for graph,seq_embed,seq_mask in loader:
            graph = graph.to(device)
            seq_embed = seq_embed.to(device)
            seq_mask = seq_mask.to(device)
          
            output = model(graph,seq_embed)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, graph.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def train_init(dataset='davis',pretrain=None):#kiba  davis
    with open('data/seq2path_prot_albert.pickle', 'rb') as handle:
            seq2path = pickle.load(handle)
    with open('data/smile_graph.pickle', 'rb') as f:
            smile2graph = pickle.load(f)
    cuda_name = CUDA
    train_data = DGSDTADataset('data/{}_train.csv'.format(dataset),smile2graph,seq2path)
    test_data = DGSDTADataset('data/{}_test.csv'.format(dataset),smile2graph,seq2path)
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    # training the model
    # device = torch.device("cpu")
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    config = DGSDTAModelConfig()
    graphNet = config['graphNet']
    model = DGSDTAModel(config).to(device)
    if pretrain:
        print("used pretrain model {}".format(pretrain))
        state_dict = torch.load(pretrain)
        model.load_state_dict(state_dict)
    # loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, device, train_loader, test_loader, loss_fn, optimizer, graphNet, dataset
import matplotlib.pyplot as plt
def learning_curve(x,y1,y2):
    # data_x, data_y = make_classification(n_samples=1000, n_classes=4, n_features=10, n_informative=8)  # 生成分类任务
    # # 绘制学习曲线
    # model = SVC(kernel="linear")
    train_sizes, train_scores, valid_scores = learning_curve(model, x, y1, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, random_state=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    # 开始绘图
    plt.plot(x, y1, label='Test MSE')
    # plt.scatter(x_data, y_data, color='red', label='Data Points')
    plt.plot(x, y2, label='Train MSE')
    # 添加标题和坐标轴标签
    plt.title("Learning Curve")
    plt.xlabel("epoch")
    plt.ylabel("MSE")

    # 显示图例
    plt.legend()

    # 显示网格线
    plt.grid(True)

    # 显示图形
    plt.show()
    plt.savefig("learning_curve_KIBA.jpg")


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE =64
# TRAIN_BATCH_SIZE = 128
# TEST_BATCH_SIZE =128
LR = 0.0002
LOG_INTERVAL = 100
NUM_EPOCHS = 1000
# CUDA = 'cuda:1'
CUDA = 'cuda:0'

# pretrain = 'model_DGSDTA_davis.model'
pretrain = False
model, device, train_loader, test_loader, loss_fn, optimizer, graphNet, dataset = train_init(pretrain=pretrain)
model_file_name = 'model_' + graphNet + '_' + dataset +  '.model'
result_file_name = 'result_' + graphNet + '_' + dataset +  '.csv'

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('The best model will be saved at: {}'.format(model_file_name))

best_mse = 1000
best_ci = 0
best_epoch = -1
for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch+1)
    G,P = predicting(model, device, test_loader)
    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
    if ret[1]<best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_ci = ret[-1]
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,graphNet,dataset)
    else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,graphNet,dataset)

