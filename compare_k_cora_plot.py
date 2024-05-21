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


clean_acc1=[0.8248663101604278, 0.8342245989304813, 0.8302139037433155, 0.8355614973262031, 0.8315508021390374, 0.8221925133689839, 0.8315508021390374, 0.8315508021390374, 0.8288770053475936, 0.8315508021390374]
attacked_acc1=[0.766042780748663, 0.7633689839572192, 0.7606951871657753, 0.7687165775401069, 0.7553475935828876, 0.767379679144385, 0.7620320855614973, 0.7593582887700534, 0.7606951871657753, 0.7620320855614973]

clean_acc2=[0.8395721925133689, 0.8409090909090908, 0.8435828877005347, 0.8529411764705882, 0.8435828877005347, 0.8516042780748663, 0.8462566844919786, 0.8435828877005347, 0.8409090909090908, 0.8462566844919786]
attacked_acc2=[0.8475935828877005, 0.8449197860962566, 0.8449197860962566, 0.8502673796791443, 0.8449197860962566, 0.8395721925133689, 0.8462566844919786, 0.8502673796791443, 0.8489304812834224, 0.8449197860962566]

clean_acc3=[0.8302139037433155, 0.8101604278074865, 0.8021390374331551, 0.820855614973262, 0.820855614973262, 0.7687165775401069, 0.7754010695187166, 0.81951871657754, 0.8302139037433155, 0.803475935828877]
attacked_acc3= [0.7112299465240641, 0.783422459893048, 0.8288770053475936, 0.8061497326203209, 0.8302139037433155, 0.6938502673796791, 0.8235294117647058, 0.8409090909090908, 0.81951871657754, 0.8275401069518716]

clean_acc4=[0.5815508021390374, 0.4572192513368984, 0.2927807486631016, 0.2927807486631016, 0.2927807486631016, 0.2927807486631016, 0.2927807486631016, 0.4371657754010695, 0.2927807486631016, 0.2927807486631016]
attacked_acc4= [0.2927807486631016, 0.2927807486631016, 0.2927807486631016, 0.5521390374331551, 0.42914438502673796, 0.2927807486631016, 0.2927807486631016, 0.2927807486631016, 0.2927807486631016, 0.2927807486631016]


data=pd.DataFrame({"C_k=2":clean_acc1,"P_k=2":attacked_acc1, 
                   "C_k=6":clean_acc2,"P_k=6":attacked_acc2, 
                    "C_k=12":clean_acc3,"P_k=12":attacked_acc3,
                    "C_k=24":clean_acc4,"P_k=24":attacked_acc4 })

#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()




