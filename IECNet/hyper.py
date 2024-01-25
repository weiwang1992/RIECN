import torch
import torch.nn
from torch.nn import functional as F


class HYPER(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(HYPER, self).__init__()
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

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))
        fc_length = (1 - self.filt_height + 1) * (ent_dim - self.filt_width + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, ent_dim)
        fc1_length = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        self.fc1 = torch.nn.Linear(rel_dim, fc1_length)

    def init(self):
        torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)

    def forward(self, entity_id, relation_id):
        e1 = self.entity_embedding(entity_id).view(-1, 1, 1, self.entity_embedding.weight.size(1))
        r = self.relation_embedding(relation_id)
        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_height, self.filt_width)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_height, self.filt_width)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 - self.filt_height + 1, e1.size(3) - self.filt_width + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)
        return pred