import torch
from gcn import GCN
from utils import *
import argparse
import numpy as np
from metattack import MetaApprox, Metattack
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from deeprobust.graph.data import Dataset, PrePtbDataset


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='pubmed',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

args = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


# === loading dataset

a=np.load("data/cora.npz")
print(a.files)

c=np.load("data/citeseer.npz")
print(c.files)


b=np.load("data/pubmed.npz")
print(b.files)




data= Dataset(root='/tmp/', name='pubmed')

print(data)


"""
adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1

print("adj",adj.shape)
print("features",features.shape)
print("labels",labels.shape)


print(adj)
"""


