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
parser.add_argument('--k', type=int, default=6, help='iteration number of HITS score.')
#parser.add_argument('--l', type=int, default=10, help='iteration number of propogation.')

parser.add_argument('--alpha', type=float, default=0.5, help='message passing parameter.')
parser.add_argument('--beta', type=float, default=0.5, help='message passing parameter.')

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


class HITSGNN_ADAPTIVE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super(HITSGNN_ADAPTIVE, self).__init__()

        self.layer1 = Encoder(nfeat, nhid, with_bias=with_bias)
        self.layer2 = Encoder(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.with_relu = with_relu

        """
    def update(self, adj, nfeat):

        #authority=torch.ones(adj.shape[0]).T.to(device)
        #hub=torch.ones(adj.shape[0]).T.to(device)

        for i in range(args.k):
            authority =  adj @ nfeat
            authority = adj.T @ authority

            hub = adj.T @ nfeat
            hub = adj @ hub

        return F.softmax(authority+hub,dim=0)"""


    def forward(self, x, adj):

        authority=x
        hub=x
        for i in range(args.k):
            authority =  adj @ x
            authority = adj.T @ authority
            #authority = adj.T @ authority

            hub = adj.T @ x
            hub = adj @ hub
            #hub = adj.T @ hub

        x=args.alpha*authority+args.beta*hub


        if self.with_relu:
            x = F.relu(self.layer1(x, adj))
        else:
            x = self.layer1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)

        #print("x_shape:",x.shape)
        #print("adj_shape:",adj.shape)



        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()


