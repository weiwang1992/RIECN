import torch
import torch.nn
from torch.nn import functional as F


class CONVE(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(CONVE, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]

        self.entity_embedding = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.relation_embedding = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels,
                            (self.filt_height, self.filt_width), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)  #归一化
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(ent_dim) # 批标准化 梯度变大，避免梯度消失加快收敛速度，加快训练速度
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))
        fc_length = (20-self.filt_height+1)*(20-self.filt_width+1)*self.out_channels
        self.fc = torch.nn.Linear(fc_length, ent_dim)

    def init(self):
        torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)


    def forward(self, e1_idx, r_idx):
        e1 = self.entity_embedding(e1_idx).view(-1, 1, 10, 20)
        r = self.relation_embedding(r_idx).view(-1, 1, 10, 20)
        x = torch.cat([e1, r], 2)#拼接
        x = self.bn0(x)# 批标准化
        x= self.inp_drop(x)# dropout  防止过拟合
        x= self.conv1(x)#卷积运算
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)# x=[num,d1]----[num,1,d1]
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.mm(x, self.entity_embedding.weight.transpose(1,0)) # 二分类输出层激活sigmoid BinaryCrossEntropy
        # 多分类输出层激活softmax CategoricalCrossEntropy
        # 多标签分类输出层激活sigmoid BinaryCrossEntropy
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)
        return pred