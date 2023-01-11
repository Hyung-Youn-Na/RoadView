import torch.nn as nn
from language.metric import grufc_accuracy
from language.base import BaseModel


class GRUFC(BaseModel):
    def __init__(self, input_dim, output_dim, gru_num=1, with_bn=False, with_bi=False, dropout=0, with_w=False):
        super(GRUFC, self).__init__()
        if with_bn:
            print('with bn')
        if with_bi:
            print('with bi_rnn')
        if dropout > 0:
            print('with dropout %f'%dropout)
        self.with_bn = with_bn
        self.with_bi = with_bi
        self.with_w = with_w
        hidden_dim = 1024
        self.rnn = nn.GRU(input_dim, hidden_dim, gru_num, bidirectional=with_bi, dropout=dropout)
        self.relu = nn.ReLU(inplace=True)
        if with_bi:
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.with_w:
            self.fcw = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        # self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        init_list = [self.fc1, self.fc2, self.fc3]
        if self.with_w:
            init_list.append(self.fcw)
        for m in init_list:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        x = x.transpose(0, 1)
        # self.rnn.flatten_parameters()
        x = self.rnn(x)
        if self.with_bi:
            x = x[1][-2:, :, :].transpose(0, 1).contiguous().view(-1, 1024 * 2)
        else:
            x = x[1][-1, :, :].view(-1, 1024)
        if self.with_bn:
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
        else:
            x = self.relu(self.fc1(x))
            if return_feat:
                feat = x
            x = self.relu(self.fc2(x))
        if self.with_w:
            w = self.fcw(x)
        x = self.fc3(x)

        ret = [x]
        if self.with_w:
            ret.append(w)
        if return_feat:
            ret.append(feat)
        return ret

    def loss(self,
             cls_scores,
             labels,
             type=''):
        losses = dict()

        suffix = type
        if len(type) > 0:
            suffix = '_' + type

        losses['loss_cls' + suffix] = self.criterion(cls_scores, labels)
        losses['acc' + suffix] = grufc_accuracy(cls_scores, labels)
        # print(losses['loss_cls' + suffix].shape)
        # exit()

        return losses