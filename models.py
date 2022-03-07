from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGraphConv(nn.Module):
    def __init__(self):
        super(LightGraphConv, self).__init__()

    def forward(self, x, A_hat):
        return A_hat.matmul(x)

class NGCF(nn.Module):
    def __init__(self, in_ch, emb_ch, dropout_rate = 0.1):
        super(NGCF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(in_ch, emb_ch)
        self.lgc1 = LightGraphConv()
        self.lgc2 = LightGraphConv()
        self.lgc3 = LightGraphConv()
        self.fc1_1 = nn.Linear(emb_ch, emb_ch, bias = True)
        self.fc1_2 = nn.Linear(emb_ch, emb_ch, bias = True)
        self.fc2_1 = nn.Linear(emb_ch, emb_ch, bias = True)
        self.fc2_2 = nn.Linear(emb_ch, emb_ch, bias = True)
        self.fc3_1 = nn.Linear(emb_ch, emb_ch, bias = True)
        self.fc3_2 = nn.Linear(emb_ch, emb_ch, bias = True)

        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.fc1_1.weight)
        torch.nn.init.xavier_uniform_(self.fc1_2.weight)
        torch.nn.init.xavier_uniform_(self.fc2_1.weight)
        torch.nn.init.xavier_uniform_(self.fc2_2.weight)
        torch.nn.init.xavier_uniform_(self.fc3_1.weight)
        torch.nn.init.xavier_uniform_(self.fc3_2.weight)
        self.fc1_1.bias.data.fill_(0.0)
        self.fc1_2.bias.data.fill_(0.0)
        self.fc2_1.bias.data.fill_(0.0)
        self.fc2_2.bias.data.fill_(0.0)
        self.fc3_1.bias.data.fill_(0.0)
        self.fc3_2.bias.data.fill_(0.0)

    def forward(self, x, A_hat):
        x = self.embed(x)
        x1 = self.lgc1(x, A_hat)
        x1 = F.dropout(x1, self.dropout_rate)
        x1 = F.leaky_relu(self.fc1_1(x1 + x) + self.fc1_2(x1*x), negative_slope=0.2)
        x2 = self.lgc2(x1, A_hat)
        x2 = F.dropout(x2, self.dropout_rate)
        x2 = F.leaky_relu(self.fc2_1(x2 + x1) + self.fc2_2(x2*x1), negative_slope=0.2)
        x3 = self.lgc3(x2, A_hat)
        x3 = F.dropout(x3, self.dropout_rate)
        x3 = F.leaky_relu(self.fc3_1(x3 + x2) + self.fc3_2(x3*x2), negative_slope=0.2)
        return torch.cat((x, x1, x2, x3), dim = 1)

class LightGCN(nn.Module):
    def __init__(self, in_ch, emb_ch, dropout_rate = 0.1):
        super(LightGCN, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(in_ch, emb_ch)
        self.lgc1 = LightGraphConv()
        self.lgc2 = LightGraphConv()
        self.lgc3 = LightGraphConv()

        torch.nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x, A_hat):
        x = self.embed(x)
        x1 = F.dropout(x, self.dropout_rate)
        x1 = self.lgc1(x1, A_hat)
        x2 = F.dropout(x1, self.dropout_rate)
        x2 = self.lgc2(x2, A_hat)
        x3 = F.dropout(x2, self.dropout_rate)
        x3 = self.lgc3(x3, A_hat)
        return (x + x1 + x2 + x3) / 4
        # return torch.cat((x, x1, x2, x3), dim = 1)

class LightGCN_2(nn.Module):
    def __init__(self, in_ch, emb_ch, dropout_rate = 0.1):
        super(LightGCN_2, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(in_ch, emb_ch)
        self.lgc1 = LightGraphConv()
        self.lgc2 = LightGraphConv()

        torch.nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x, A_hat):
        x = self.embed(x)
        x1 = F.dropout(x, self.dropout_rate)
        x1 = self.lgc1(x1, A_hat)
        x2 = F.dropout(x1, self.dropout_rate)
        x2 = self.lgc2(x2, A_hat)
        return (x + x1 + x2) / 3
        # return torch.cat((x, x1, x2), dim = 1)

class HG_conv(nn.Module):
    def __init__(self):
        super(HG_conv, self).__init__()

    def forward(self, x: torch.Tensor, G):
        # x.shape = (n_item, hid_ch), G.shape = (n_item, n_item)
        x = G.matmul(x)
        return x

class DHCF_1(nn.Module):
    def __init__(self, in_ch, hid_ch, i_emb, dropout_rate = 0.1):
        super(DHCF_1, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(in_ch, i_emb)
        self.hgc1 = HG_conv()
        self.fc1 = nn.Linear(i_emb, i_emb, bias = True)
        # weight initialize는 DHCF 논문 따라서
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.0)

    def forward(self, x, G):                 
        x = self.embed(x)                     
        m1 = self.hgc1(x, G) + x 
        m1 = F.dropout(m1, self.dropout_rate)
        x1 = F.leaky_relu(self.fc1(m1), negative_slope=0.2) 
        return torch.cat((x, x1), dim = 1)

class DHCF_2(nn.Module):
    def __init__(self, in_ch, hid_ch, i_emb, dropout_rate = 0.1):
        super(DHCF_2, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(in_ch, hid_ch)
        self.hgc1 = HG_conv()
        self.hgc2 = HG_conv()
        self.fc1 = nn.Linear(hid_ch, hid_ch, bias = True)
        self.fc2 = nn.Linear(hid_ch, i_emb, bias = True)
        # weight initialize는 DHCF 논문 따라서
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x,  G):                 # (x.shape = (n_item))
        x = self.embed(x)                     # (x.shape = (n_item, hid_ch))
        m1 = self.hgc1(x, G) + x # phase 1    # (m1.shape = (n_item, hid_ch))
        m1 = F.dropout(m1, self.dropout_rate)
        x1 = F.leaky_relu(self.fc1(m1), negative_slope=0.2) # phase 2   # (x1.shape = (n_item, hid_ch))
        m2 = self.hgc2(x1, G) + x1 # phase 1  # (m2.shape = (n_item, hid_ch))
        m2 = F.dropout(m2, self.dropout_rate)
        x2 = F.leaky_relu(self.fc2(m2), negative_slope=0.2) # phase 2   # (x2.shape = (n_item, i_emb))
        return torch.cat((x, x1, x2), dim = 1)

class DHCF_3(nn.Module):
    def __init__(self, in_ch, hid_ch, i_emb, dropout_rate = 0.1):
        super(DHCF_3, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(in_ch, hid_ch)
        self.hgc1 = HG_conv()
        self.hgc2 = HG_conv()
        self.hgc3 = HG_conv()
        self.fc1 = nn.Linear(hid_ch, hid_ch, bias = True)
        self.fc2 = nn.Linear(hid_ch, hid_ch, bias = True)
        self.fc3 = nn.Linear(hid_ch, i_emb, bias = True)
        # weight initialize는 DHCF 논문 따라서
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, x,  G):                 # (x.shape = (n_item))
        x = self.embed(x)                     # (x.shape = (n_item, hid_ch))
        m1 = self.hgc1(x, G) + x # phase 1    # (m1.shape = (n_item, hid_ch))
        m1 = F.dropout(m1, self.dropout_rate)
        x1 = F.leaky_relu(self.fc1(m1), negative_slope=0.2) # phase 2   # (x1.shape = (n_item, hid_ch))
        m2 = self.hgc2(x1, G) + x1 # phase 1  # (m2.shape = (n_item, hid_ch))
        m2 = F.dropout(m2, self.dropout_rate)
        x2 = F.leaky_relu(self.fc2(m2), negative_slope=0.2) # phase 2   # (x2.shape = (n_item, i_emb))
        m3 = self.hgc3(x2, G) + x2
        m3 = F.dropout(m3, self.dropout_rate)
        x3 = F.leaky_relu(self.fc3(m3), negative_slope=0.2)
        return torch.cat((x, x1, x2, x3), dim = 1)