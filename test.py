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


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='citeseer',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

args = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


clean_acc = [0.7209715639810427, 0.7251184834123223, 0.7191943127962086, 0.7203791469194314, 0.7239336492890995, 0.7209715639810427, 0.7191943127962086, 0.7251184834123223, 0.7239336492890995, 0.7280805687203792] 

attacked_acc = [0.6469194312796209, 0.6404028436018958, 0.6469194312796209, 0.6463270142180095, 0.6415876777251185, 0.648696682464455, 0.6492890995260664, 0.648696682464455, 0.6469194312796209, 0.6469194312796209]


#clean_acc=list(np.float_(clean_acc))

#attacked_acc=list(np.float_(attacked_acc))


data=pd.DataFrame({"Acc. Clean":clean_acc,"Acc. Perturbed":attacked_acc})

plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges using model {}".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()

