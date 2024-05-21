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
parser.add_argument('--dataset', type=str, default='polblogs',
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


clean_acc1=[0.9483695652173912, 0.9402173913043478, 0.9483695652173912, 0.9510869565217391, 0.9510869565217391, 0.9510869565217391, 0.9510869565217391, 0.9510869565217391, 0.9456521739130435, 0.9456521739130435]
attacked_acc1=[0.9157608695652174, 0.9157608695652174, 0.921195652173913, 0.9157608695652174, 0.9184782608695652, 0.9239130434782609, 0.9239130434782609, 0.9239130434782609, 0.9266304347826086, 0.9184782608695652]

clean_acc2=[0.9538043478260869, 0.904891304347826, 0.9402173913043478, 0.9021739130434783, 0.9619565217391304, 0.9456521739130435, 0.9239130434782609, 0.9592391304347826, 0.9510869565217391, 0.9347826086956521]
attacked_acc2=[0.8777173913043478, 0.8777173913043478, 0.8804347826086957, 0.8777173913043478, 0.8777173913043478, 0.875, 0.8831521739130435, 0.8885869565217391, 0.875, 0.8777173913043478]

clean_acc3=[0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.483695652173913, 0.5217391304347826]
attacked_acc3= [0.9375, 0.8913043478260869, 0.5217391304347826, 0.9103260869565217, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.8831521739130435, 0.5217391304347826, 0.8913043478260869]

clean_acc4=[0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826]
attacked_acc4= [0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826, 0.5217391304347826]


data=pd.DataFrame({"C_k=2":clean_acc1,"P_k=2":attacked_acc1, 
                   "C_k=6":clean_acc2,"P_k=6":attacked_acc2, 
                    "C_k=12":clean_acc3,"P_k=12":attacked_acc3,
                    "C_k=24":clean_acc4,"P_k=24":attacked_acc4 })

#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()




