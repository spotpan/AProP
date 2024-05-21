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


clean_acc1=[0.9510869565217391, 0.9347826086956521, 0.9429347826086957, 0.9320652173913043, 0.9320652173913043, 0.9402173913043478, 0.9483695652173912, 0.9266304347826086, 0.9375, 0.9510869565217391]
attacked_acc1=[0.904891304347826, 0.8967391304347826, 0.8967391304347826, 0.9130434782608695, 0.8967391304347826, 0.8940217391304348, 0.904891304347826, 0.9076086956521738, 0.9130434782608695, 0.9076086956521738]

clean_acc2=[0.9456521739130435, 0.9021739130434783, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9592391304347826, 0.9130434782608695, 0.9619565217391304, 0.9538043478260869, 0.9320652173913043]
attacked_acc2=[0.8722826086956521, 0.8668478260869565, 0.8831521739130435, 0.875, 0.875, 0.8695652173913043, 0.8831521739130435, 0.875, 0.8804347826086957, 0.8804347826086957]

clean_acc3=[0.9565217391304348, 0.8994565217391304, 0.9619565217391304, 0.9483695652173912, 0.9347826086956521, 0.9619565217391304, 0.9266304347826086, 0.9375, 0.9592391304347826, 0.9347826086956521]
attacked_acc3= [0.875, 0.8668478260869565, 0.8804347826086957, 0.875, 0.8641304347826086, 0.8777173913043478, 0.8668478260869565, 0.8695652173913043, 0.8695652173913043, 0.8858695652173912]

clean_acc4=[0.9293478260869565, 0.9293478260869565, 0.9239130434782609, 0.9320652173913043, 0.9347826086956521, 0.9320652173913043, 0.9293478260869565, 0.9320652173913043, 0.9375, 0.9320652173913043]
attacked_acc4= [0.9103260869565217, 0.904891304347826, 0.9103260869565217, 0.9076086956521738, 0.9076086956521738, 0.9076086956521738, 0.9076086956521738, 0.904891304347826, 0.9076086956521738, 0.9103260869565217]

clean_acc5=[0.9429347826086957, 0.9483695652173912, 0.9402173913043478, 0.9157608695652174, 0.9565217391304348, 0.9402173913043478, 0.9266304347826086, 0.9538043478260869, 0.9510869565217391, 0.9239130434782609]
attacked_acc5= [0.8668478260869565, 0.8668478260869565, 0.8695652173913043, 0.8695652173913043, 0.8641304347826086, 0.8722826086956521, 0.8614130434782609, 0.8668478260869565, 0.8614130434782609, 0.8695652173913043]

clean_acc6=[0.9483695652173912, 0.9076086956521738, 0.9538043478260869, 0.9375, 0.9592391304347826, 0.9592391304347826, 0.9076086956521738, 0.9592391304347826, 0.9592391304347826, 0.9320652173913043]
attacked_acc6= [0.8695652173913043, 0.8505434782608695, 0.8722826086956521, 0.8668478260869565, 0.8695652173913043, 0.8695652173913043, 0.8668478260869565, 0.8722826086956521, 0.8641304347826086, 0.8668478260869565]

clean_acc7=[0.9538043478260869, 0.904891304347826, 0.9402173913043478, 0.9021739130434783, 0.9619565217391304, 0.9456521739130435, 0.9239130434782609, 0.9592391304347826, 0.9510869565217391, 0.9347826086956521]
attacked_acc7= [0.8777173913043478, 0.8777173913043478, 0.8804347826086957, 0.8777173913043478, 0.8777173913043478, 0.875, 0.8831521739130435, 0.8885869565217391, 0.875, 0.8777173913043478]




data=pd.DataFrame({"C_0-1":clean_acc1,"P_0-1":attacked_acc1,  
                   "C_.2-.8":clean_acc2,"P_.2-.8":attacked_acc2, 
                   "C_.4-.6":clean_acc3,"P_.4-.6":attacked_acc3, 
                    "C_1-0":clean_acc4,"P_1-0":attacked_acc4,
                    "C_.8-.2":clean_acc5,"P_.8-.2":attacked_acc5, 
                    "C_.6-.4":clean_acc6,"P_.6-.4":attacked_acc6, 
                    "C_.5-.5":clean_acc7,"P_.5-.5":attacked_acc7, })

"""
#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()
"""
fig, ax = plt.subplots()

sns.boxplot(data=data, palette="husl", ax=ax)

# Get the unique values on x-axis and their corresponding positions
x_values = np.arange(len(data.columns))

# Set the background color of the figure based on the variable on x-axis
for x_position in x_values:
    col_name = data.columns[x_position]
    color = 'white' if col_name in ["C_0-1", "P_0-1","C_.4-.6","P_.4-.6","C_.8-.2","P_.8-.2",
            "C_.5-.5", "P_.5-.5"] else (0.8, 0.8, 0.8)  # Light gray
    ax.axvspan(x_position - 0.5, x_position + 0.5, facecolor=color, alpha=0.3)

# Set x-axis labels
ax.set_xticks(x_values)
ax.set_xticklabels(data.columns, rotation=45, ha='right')

# Show the plot
#plt.tight_layout()
#plt.show()

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("prop_rules/results_on_{}.png".format(args.dataset), dpi=600)
plt.show()



