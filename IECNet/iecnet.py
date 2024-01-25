import torch
import torch.nn
from torch.nn import functional as F


class IECNET(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(IECNET, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_height = 20
        self.reshape_width = 20
        self.out = 16


        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]

        self.entity_embedding = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.relation_embedding = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        filter_dim = self.in_channels * self.out * 3 * 3
        self.filter = torch.nn.Embedding(data.relations_num, filter_dim, padding_idx=0)
        filter1_dim = self.in_channels * self.out * 1 * 9
        self.filter1 = torch.nn.Embedding(data.relations_num, filter1_dim, padding_idx=0)

        self.input_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)

        self.bn1_1 = torch.nn.BatchNorm2d(self.out)

        self.bn2 = torch.nn.BatchNorm1d(ent_dim)

        self.loss = torch.nn.BCELoss()
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))


        fc_length = self.reshape_height * self.reshape_width * self.out*2
        self.fc = torch.nn.Linear(fc_length, ent_dim)

    def init(self):
        torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.filter.weight.data)
        torch.nn.init.xavier_normal_(self.filter1.weight.data)


    def forward(self, entity_id, relation_id):
        # (b, 1, 200)
        entity = self.entity_embedding(entity_id)
        relation = self.relation_embedding(relation_id)
        f = self.filter(relation_id)
        f = f.reshape(entity.size(0) * self.in_channels * self.out, 1, 3, 3)
        f1 = self.filter1(relation_id)
        f1 = f1.reshape(entity.size(0) * self.in_channels * self.out, 1, 1, 9)
        # (b, 2, 200)→ (b, 200, 2)→ (b, 1, 20, 20)
        x = torch.cat([entity, relation], 1).reshape(-1,1,10,40)

        list = torch.chunk(x, dim=3, chunks=4)
        #a, b, c, d = torch.chunk(x, 4, dim=3)
        t1 = list[1].transpose(2, 3)
        t2 = list[3].transpose(2, 3)
        m = torch.cat([list[0], list[2]], 2)
        n = torch.cat([t2, t1], 2)
        x = torch.cat([m, n], 3)


        x = self.bn0(x)
        x = self.input_drop(x)

        # (1 ,b, 20, 20)
        x = x.permute(1, 0, 2, 3)



        x1 = F.conv2d(x, f, groups=entity.size(0), padding=(1,1))
        x1= x1.reshape(entity.size(0), self.out, self.reshape_height, self.reshape_width)
        x1 = self.bn1_1(x1)
        x2 = F.conv2d(x, f1, groups=entity.size(0), padding=(0, 4))
        x2 = x2.reshape(entity.size(0), self.out, self.reshape_height, self.reshape_width)
        x2 = self.bn1_1(x2)

        x = torch.cat([x1,x2],dim=1)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # (b, fc_length)
        x = x.view(entity.size(0), -1)

        # (b, ent_dim)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # (batch, ent_dim)*(ent_dim, ent_num)=(batch, ent_num)
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
