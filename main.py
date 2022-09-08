import random

from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn.conv import GINEConv
import numpy as np
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels, hid_channels, t=1.):
        super(Model, self).__init__()

        mlp = nn.Sequential(nn.Linear(in_channels, hid_channels),
                            nn.ReLU(),
                            nn.Linear(hid_channels, hid_channels)
                            )

        self.t = t

        self.gine_conv = GINEConv(nn=mlp, eps=0., train_eps=False, edge_dim=64)
        self.thresholding_node = nn.Parameter(torch.Tensor(1, hid_channels))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.thresholding_node)

    def forward(self, x, edge_index, edge_attr, pos_ind, neg_ind):
        """ 训练用函数  输入为异常子图和采样得到的正负邻居节点的index
        返回训练loss用于优化
        :param x:
        :param edge_index:
        :param edge_feat:
        :param pos_ind:
        :param neg_ind:
        :return:
        """
        emb = self.gine_conv(x, edge_index, edge_attr)

        center_node = emb[0]
        pos_neigh = emb[pos_ind]
        neg_neigh = emb[neg_ind]

        pos_neigh = F.normalize(pos_neigh, dim=-1, eps=1e-12)
        neg_neigh = F.normalize(neg_neigh, dim=-1, eps=1e-12)
        center_node = F.normalize(center_node, dim=-1, eps=1e-12)
        thresholding_node = F.normalize(self.thresholding_node, dim=-1, eps=1e-12)

        pos_score = (center_node.unsqueeze(dim=0) * pos_neigh).sum(-1)
        neg_score = (center_node.unsqueeze(dim=0) * neg_neigh).sum(-1)

        thresholding_node_score = (center_node * thresholding_node).sum(-1)

        pos_score_mean = pos_score.mean(dim=-1)
        neg_score_mean = neg_score.mean(dim=-1)

        pos_loss = torch.sigmoid((pos_score_mean - thresholding_node_score) / self.t)
        neg_loss = torch.sigmoid((thresholding_node_score - neg_score_mean) / self.t)

        pos_loss = torch.log(pos_loss.clamp(min=1e-12)).mean()
        neg_loss = torch.log(neg_loss.clamp(min=1e-12)).mean()

        loss = 0.
        loss -= pos_loss
        loss -= neg_loss

        return loss

    @torch.no_grad()
    def get_critical_subgraph(self, x, edge_index, edge_attr):
        """
        输入为节点子图 输出为关键节点index set
        :param x:
        :param edge_index:
        :param edge_attr:
        :return:
        """
        emb = self.gine_conv(x, edge_index, edge_attr)
        center_node = emb[0]

        neigh = F.normalize(emb, dim=-1, eps=1e-12)
        center_node = F.normalize(center_node, dim=-1, eps=1e-12)
        thresholding_node = F.normalize(self.thresholding_node, dim=-1, eps=1e-12)

        neigh_score = (center_node.unsqueeze(dim=0) * neigh).sum(-1)
        thresholding_node_score = (center_node * thresholding_node).sum(-1)

        sub_node_mask = neigh_score > thresholding_node_score
        sub_node_index = sub_node_mask.nonzero(as_tuple=False).view(-1)

        return sub_node_index


if __name__ == '__main__':
    # 节点二分类任务 index为0为中心节点
    fake_dataset = FakeDataset(num_graphs=512, avg_num_nodes=100, avg_degree=10, num_channels=64, edge_dim=64,
                num_classes=2, task='node')

    for g in fake_dataset:
        g.y[0] = 1

    dataloader = DataLoader(dataset=fake_dataset, batch_size=1, shuffle=True)

    # 正负采样数
    positive_num = 8
    negative_num = 16

    model = Model(in_channels=64, hid_channels=64)


    for i, g in enumerate(dataloader):
        positive_index = g.y.bool().nonzero(as_tuple=False).view(-1)
        negative_index = (~(g.y.bool())).nonzero(as_tuple=False).view(-1)

        pos_ind = positive_index[random.sample(range(len(positive_index)), positive_num)]
        neg_ind = negative_index[random.sample(range(len(negative_index)), negative_num)]

        ## TODO: 优化以下Loss
        model.train()

        loss = model(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, pos_ind=pos_ind, neg_ind=neg_ind)

    test_g = fake_dataset[0]
    model.eval()
    subset_node = model.get_critical_subgraph(x=test_g.x, edge_index=test_g.edge_index, edge_attr=test_g.edge_attr)
    d = 1