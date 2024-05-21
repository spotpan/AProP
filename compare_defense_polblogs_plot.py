import torch
#from hitsgnn import HITSGNN
from hits_prgnn import HITSGNN
from hits_prgnn_chameleon import HITSGNN_ADAPTIVE
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
parser.add_argument('--dataset', type=str, default='cora',
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


clean_acc1=[0.9347826086956521, 0.9456521739130435, 0.9375, 0.9429347826086957, 0.9510869565217391, 0.9456521739130435, 0.9483695652173912, 0.9402173913043478, 0.9510869565217391, 0.9483695652173912]
attacked_acc1=[0.9293478260869565, 0.9157608695652174, 0.9266304347826086, 0.921195652173913, 0.9184782608695652, 0.9293478260869565, 0.9239130434782609, 0.9239130434782609, 0.921195652173913, 0.9347826086956521]
defense_acc1=[0.921195652173913, 0.9266304347826086, 0.9239130434782609, 0.921195652173913, 0.921195652173913, 0.9266304347826086, 0.9266304347826086, 0.9239130434782609, 0.9157608695652174, 0.9184782608695652]

clean_acc2=[0.9293478260869565, 0.9375, 0.9293478260869565, 0.9320652173913043, 0.9347826086956521, 0.9320652173913043, 0.9266304347826086, 0.9320652173913043, 0.9320652173913043, 0.9320652173913043]
attacked_acc2=[0.8967391304347826, 0.8940217391304348, 0.8940217391304348, 0.8858695652173912, 0.8940217391304348, 0.8885869565217391, 0.8967391304347826, 0.8940217391304348, 0.8940217391304348, 0.8967391304347826]
defense_acc2=[0.8831521739130435, 0.904891304347826, 0.9021739130434783, 0.9021739130434783, 0.904891304347826, 0.9076086956521738, 0.9130434782608695, 0.9076086956521738, 0.904891304347826, 0.9103260869565217]

clean_acc3=[0.9510869565217391, 0.9347826086956521, 0.9429347826086957, 0.9320652173913043, 0.9320652173913043, 0.9402173913043478, 0.9483695652173912, 0.9266304347826086, 0.9375, 0.9510869565217391]
attacked_acc3= [0.904891304347826, 0.8967391304347826, 0.8967391304347826, 0.9130434782608695, 0.8967391304347826, 0.8940217391304348, 0.904891304347826, 0.9076086956521738, 0.9130434782608695, 0.9076086956521738]
defense_acc3=[0.8967391304347826, 0.921195652173913, 0.8967391304347826, 0.9130434782608695, 0.8940217391304348, 0.9266304347826086, 0.9130434782608695, 0.9184782608695652, 0.9157608695652174, 0.921195652173913]

data=pd.DataFrame({"P_C":clean_acc1,"P_P":attacked_acc1, "P_D":defense_acc1, 
                   "H_C":clean_acc2,"H_P":attacked_acc2, "H_D":defense_acc2,
                    "F_C":clean_acc3,"F_P":attacked_acc3, "F_D":defense_acc3 })

#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()




