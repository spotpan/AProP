import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import argparse
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='iteration number of HITS score.')
parser.add_argument('--l', type=int, default=10, help='iteration number of propogation.')

args = parser.parse_args()

class Encoder(Module):


    def __init__(self, in_features, out_features, with_bias=True):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HITSGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super(HITSGNN, self).__init__()

        self.layer1 = Encoder(nfeat, nhid, with_bias=with_bias)
        self.layer2 = Encoder(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.with_relu = with_relu


    def hits_score(self, adj):

        authority=torch.ones(adj.shape[0]).T.to(device)
        hub=torch.ones(adj.shape[0]).T.to(device)

        for i in range(args.k):
            authority =  adj @ authority
            authority = adj.T @ authority

            hub = adj.T @ hub
            hub = adj @ hub

        return F.softmax(authority+hub,dim=0)


    def forward(self, x, adj):



        if self.with_relu:
            x = F.relu(self.layer1(x, adj))
        else:
            x = self.layer1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)

        #print("x_shape:",x.shape)
        #print("adj_shape:",adj.shape)

        hits=self.hits_score(adj)


        #print("hits:",hits)
        #print("hits_shape:",hits.shape)
        p=x
        beta=0.7

        for i in range(args.l):

            x = (1-beta) * hits @ x + beta * p

        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()


