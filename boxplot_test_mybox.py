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


clean_acc1=[0.7383177570093458, 0.7476635514018691, 0.7616822429906541, 0.7570093457943925, 0.7570093457943925, 0.7476635514018691, 0.7523364485981308, 0.7523364485981308, 0.7429906542056074, 0.7429906542056074]
attacked_acc1=[0.6448598130841121, 0.6495327102803737, 0.6448598130841121, 0.6401869158878504, 0.6401869158878504, 0.6448598130841121, 0.6261682242990654, 0.6401869158878504, 0.6588785046728971, 0.6448598130841121]

clean_acc2=[0.7523364485981308, 0.7616822429906541, 0.7476635514018691, 0.7570093457943925, 0.7663551401869159, 0.733644859813084, 0.7523364485981308, 0.7523364485981308, 0.7429906542056074, 0.7616822429906541]
attacked_acc2=[0.7476635514018691, 0.7616822429906541, 0.7663551401869159, 0.7616822429906541, 0.7663551401869159, 0.7710280373831775, 0.7570093457943925, 0.7710280373831775, 0.7523364485981308, 0.7616822429906541]

clean_acc3=[0.7663551401869159, 0.7710280373831775, 0.7710280373831775, 0.7663551401869159, 0.7757009345794392, 0.7383177570093458, 0.7803738317757009, 0.7616822429906541, 0.7616822429906541, 0.7803738317757009]
attacked_acc3= [0.7616822429906541, 0.7710280373831775, 0.7523364485981308, 0.7710280373831775, 0.7710280373831775, 0.7757009345794392, 0.7663551401869159, 0.7850467289719626, 0.7616822429906541, 0.7663551401869159]

clean_acc4=[0.7429906542056074, 0.7429906542056074, 0.7383177570093458, 0.7383177570093458, 0.7289719626168224, 0.7429906542056074, 0.733644859813084, 0.7570093457943925, 0.7383177570093458, 0.7383177570093458]
attacked_acc4= [0.7663551401869159, 0.7289719626168224, 0.7476635514018691, 0.7429906542056074, 0.7523364485981308, 0.7383177570093458, 0.7476635514018691, 0.7570093457943925, 0.7383177570093458, 0.7476635514018691]

clean_acc5=[0.7616822429906541, 0.7663551401869159, 0.7523364485981308, 0.7616822429906541, 0.7616822429906541, 0.7570093457943925, 0.7616822429906541, 0.7710280373831775, 0.7523364485981308, 0.7570093457943925]
attacked_acc5= [0.6542056074766355, 0.6542056074766355, 0.6588785046728971, 0.6682242990654205, 0.6542056074766355, 0.6588785046728971, 0.6635514018691588, 0.6635514018691588, 0.6728971962616822, 0.6682242990654205]

clean_acc6=[0.7616822429906541, 0.7570093457943925, 0.7663551401869159, 0.7710280373831775, 0.7710280373831775, 0.7616822429906541, 0.7710280373831775, 0.7476635514018691, 0.7570093457943925, 0.7757009345794392]
attacked_acc6= [0.7570093457943925, 0.7897196261682242, 0.7523364485981308, 0.7523364485981308, 0.7523364485981308, 0.7663551401869159, 0.7710280373831775, 0.7710280373831775, 0.7710280373831775, 0.7570093457943925]

clean_acc7=[0.7850467289719626, 0.7616822429906541, 0.7663551401869159, 0.7616822429906541, 0.7803738317757009, 0.7523364485981308, 0.7663551401869159, 0.7710280373831775, 0.7476635514018691, 0.7663551401869159]
attacked_acc7= [0.7663551401869159, 0.7570093457943925, 0.7429906542056074, 0.7570093457943925, 0.7429906542056074, 0.7710280373831775, 0.7570093457943925, 0.7757009345794392, 0.7616822429906541, 0.7570093457943925]

"""
data=pd.DataFrame({ 
    "prop":["C_0-1", "P_0-1","C_.2-.8","P_.2-.8","C_.4-.6","P_.4-.6","C_1-0","P_1-0", "C_.8-.2","P_.8-.2",
            "C_.6-.4", "P_.6-.4", "C_.5-.5", "P_.5-.5"],
    "acc": [clean_acc1,attacked_acc1,  
            clean_acc2,attacked_acc2, 
            clean_acc3,attacked_acc3, 
            clean_acc4,attacked_acc4,
            clean_acc5,attacked_acc5, 
            clean_acc6,attacked_acc6, 
            clean_acc7,attacked_acc7] })


"""


data=pd.DataFrame({"C_0-1":clean_acc1,"P_0-1":attacked_acc1,  
                   "C_.2-.8":clean_acc2,"P_.2-.8":attacked_acc2, 
                   "C_.4-.6":clean_acc3,"P_.4-.6":attacked_acc3, 
                    "C_1-0":clean_acc4,"P_1-0":attacked_acc4,
                    "C_.8-.2":clean_acc5,"P_.8-.2":attacked_acc5, 
                    "C_.6-.4":clean_acc6,"P_.6-.4":attacked_acc6, 
                    "C_.5-.5":clean_acc7,"P_.5-.5":attacked_acc7, })

#plt.figure(figsize=(6,6))
#sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])


# Create a figure
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




