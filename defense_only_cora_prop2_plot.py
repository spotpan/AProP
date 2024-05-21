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


clean_acc1= [0.836898395721925, 0.838235294117647, 0.8422459893048128, 0.8355614973262031, 0.8328877005347594, 0.8355614973262031, 0.8409090909090908, 0.8355614973262031, 0.8409090909090908, 0.8422459893048128]
attacked_acc1=[0.8008021390374331, 0.7967914438502673, 0.7954545454545454, 0.7954545454545454, 0.7927807486631016, 0.8048128342245989, 0.8008021390374331, 0.7967914438502673, 0.8048128342245989, 0.8008021390374331]

clean_acc2=[0.8409090909090908, 0.8462566844919786, 0.8355614973262031, 0.8328877005347594, 0.8462566844919786, 0.8315508021390374, 0.8489304812834224, 0.8355614973262031, 0.8355614973262031, 0.8475935828877005]
attacked_acc2=[0.7941176470588235, 0.7914438502673796, 0.7794117647058824, 0.7874331550802138, 0.783422459893048, 0.7954545454545454, 0.7794117647058824, 0.7914438502673796, 0.7914438502673796, 0.7887700534759358]

clean_acc3=[0.836898395721925, 0.8462566844919786, 0.8342245989304813, 0.8328877005347594, 0.8489304812834224, 0.8315508021390374, 0.8395721925133689, 0.8435828877005347, 0.8449197860962566, 0.8449197860962566]
attacked_acc3=[0.783422459893048, 0.7860962566844919, 0.7780748663101604, 0.7780748663101604, 0.7740641711229946, 0.78475935828877, 0.7780748663101604, 0.7767379679144385, 0.7767379679144385, 0.7767379679144385]

clean_acc4=[0.836898395721925, 0.836898395721925, 0.8489304812834224, 0.8395721925133689, 0.8516042780748663, 0.838235294117647, 0.838235294117647, 0.8422459893048128, 0.8302139037433155, 0.8395721925133689]
attacked_acc4= [0.7994652406417112, 0.7954545454545454, 0.7954545454545454, 0.8061497326203209, 0.7981283422459893, 0.7994652406417112, 0.8021390374331551, 0.7941176470588235, 0.7914438502673796, 0.7927807486631016]

clean_acc5=[0.836898395721925, 0.8409090909090908, 0.8355614973262031, 0.838235294117647, 0.8529411764705882, 0.836898395721925, 0.8395721925133689, 0.8489304812834224, 0.836898395721925, 0.8435828877005347]
attacked_acc5= [0.7941176470588235, 0.7967914438502673, 0.7901069518716577, 0.7941176470588235, 0.7887700534759358, 0.7994652406417112, 0.7927807486631016, 0.7820855614973261, 0.7914438502673796, 0.7927807486631016]

clean_acc6=[0.8395721925133689, 0.8435828877005347, 0.8342245989304813, 0.838235294117647, 0.8475935828877005, 0.8328877005347594, 0.8409090909090908, 0.8422459893048128, 0.8355614973262031, 0.8395721925133689]
attacked_acc6= [0.7767379679144385, 0.7860962566844919, 0.7794117647058824, 0.7767379679144385, 0.7767379679144385, 0.7820855614973261, 0.7820855614973261, 0.7740641711229946, 0.7754010695187166, 0.7754010695187166]

clean_acc7=[0.838235294117647, 0.8449197860962566, 0.838235294117647, 0.8328877005347594, 0.8516042780748663, 0.8315508021390374, 0.8462566844919786, 0.8435828877005347, 0.836898395721925, 0.8409090909090908]
attacked_acc7=[0.7860962566844919, 0.78475935828877, 0.783422459893048, 0.7754010695187166, 0.7807486631016042, 0.7767379679144385, 0.783422459893048, 0.7807486631016042, 0.7860962566844919, 0.783422459893048]



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


