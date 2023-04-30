import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GAT_GCN(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GAT_GCN,self).__init__()
        self.conv1 = GATConv(graph_features_dim, graph_features_dim, heads=10)
        self.conv2 = GCNConv(graph_features_dim*10, graph_features_dim*10)
        self.fc_g1 = torch.nn.Linear(graph_features_dim*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GATNet,self).__init__() 
        self.gcn1 = GATConv(graph_features_dim, graph_features_dim, heads=10, dropout=dropout)
        self.gcn2 = GATConv(graph_features_dim * 10, graph_output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(graph_output_dim, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        return x 

class SeqNet(torch.nn.Module):
    def __init__(self,seq_embed_dim=1024,n_filters=256,seq_output_dim=1024,dropout=0.2):
        super(SeqNet,self).__init__()
        
        self.conv_xt_1 = nn.Conv1d(in_channels=seq_embed_dim, out_channels=n_filters, kernel_size=5)
        self.pool_xt_1 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters,out_channels=seq_embed_dim,kernel_size=5)
        self.pool_xt_2 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.conv_xt_3 = nn.Conv1d(in_channels=seq_embed_dim,out_channels=n_filters,kernel_size=5)
        self.pool_xt_3 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.conv_xt_4 = nn.Conv1d(in_channels=n_filters,out_channels=int(n_filters/2),kernel_size=3)
        self.pool_xt_4 = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.fc1_xt = nn.Linear(128*61, seq_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
   
    def forward(self,seq_embed,seq_mask=None):
        # 1d conv layers
        xt = self.conv_xt_1(seq_embed.transpose(1,2))
        xt = self.relu(xt)
        xt = self.pool_xt_1(xt)
        xt = self.conv_xt_2(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_2(xt)
        xt = self.conv_xt_3(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_3(xt)
        xt = self.conv_xt_4(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_4(xt)

        # flatten
        xt = xt.view(-1, 128*61)
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        return xt

class DGSDTA(torch.nn.Module):
    def __init__(self,config):
        super(DGSDTA, self).__init__()
        dropout = config['dropout']
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # SMILES graph branch
        if config['graphNet'] == 'GAT_GCN':
            self.graph = GAT_GCN(config['graph_output_dim'],config['graph_features_dim'],dropout)
        elif config['graphNet'] == 'GAT':
            self.graph = GATNet(config['graph_output_dim'],config['graph_features_dim'],dropout)
        else:
            print("Unknow model name")
        
        # Seq branch
        self.seqnet = SeqNet(config['seq_embed_dim'],config['n_filters'],config['seq_output_dim'],dropout)

        # combined layers
        self.fc1 = nn.Linear(config['graph_output_dim']+config['seq_output_dim'], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)
        # self.bilstm = nn.LSTM(64, 64, num_layers=1, bidirectional=True, dropout=self.dropout)

    def forward(self, graph,seq_embed,seq_mask=None):
        graph_output = self.graph(graph)
        seq_output = self.seqnet(seq_embed,seq_mask)
        # concat
        h_0 = Variable(torch.zeros(2, 1, 64).to(device))
        c_0 = Variable(torch.zeros(2, 1, 64).to(device))
        #
        seq_output, _ = self.bilstm(seq_output, (h_0, c_0))
        xc = torch.cat((graph_output, seq_output), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        out = self.out(xc)
        return out
