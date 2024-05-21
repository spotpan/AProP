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


clean_acc1=[0.8422459893048128, 0.8449197860962566, 0.8475935828877005, 0.8462566844919786, 0.8502673796791443, 0.8435828877005347, 0.8409090909090908, 0.8435828877005347, 0.8435828877005347, 0.8475935828877005]
attacked_acc1=[0.78475935828877, 0.7874331550802138, 0.7887700534759358, 0.7860962566844919, 0.7780748663101604, 0.78475935828877, 0.7794117647058824, 0.7807486631016042, 0.7874331550802138, 0.7807486631016042]
defense_acc1=[0.7740641711229946, 0.78475935828877, 0.7807486631016042, 0.7754010695187166, 0.78475935828877, 0.7780748663101604, 0.7767379679144385, 0.7767379679144385, 0.7807486631016042, 0.7767379679144385]

clean_acc2=[0.836898395721925, 0.838235294117647, 0.8422459893048128, 0.8355614973262031, 0.8328877005347594, 0.8355614973262031, 0.8409090909090908, 0.8355614973262031, 0.8409090909090908, 0.8422459893048128]
attacked_acc2=[0.8008021390374331, 0.7967914438502673, 0.7954545454545454, 0.7954545454545454, 0.7927807486631016, 0.8048128342245989, 0.8008021390374331, 0.7967914438502673, 0.8048128342245989, 0.8008021390374331]
defense_acc2=[0.8048128342245989, 0.7967914438502673, 0.8061497326203209, 0.8061497326203209, 0.8048128342245989, 0.7967914438502673, 0.8021390374331551, 0.8061497326203209, 0.8048128342245989, 0.8114973262032085]

clean_acc3=[0.838235294117647, 0.8409090909090908, 0.8395721925133689, 0.8302139037433155, 0.8409090909090908, 0.8422459893048128, 0.8342245989304813, 0.8315508021390374, 0.8409090909090908, 0.8342245989304813]
attacked_acc3= [0.8409090909090908, 0.8235294117647058, 0.836898395721925, 0.8275401069518716, 0.8302139037433155, 0.8315508021390374, 0.8288770053475936, 0.836898395721925, 0.8288770053475936, 0.8342245989304813]
defense_acc3=[0.836898395721925, 0.836898395721925, 0.8395721925133689, 0.8342245989304813, 0.8302139037433155, 0.8449197860962566, 0.8342245989304813, 0.8315508021390374, 0.838235294117647, 0.8328877005347594]

data=pd.DataFrame({"P_C":clean_acc1,"P_P":attacked_acc1, "P_D":defense_acc1, 
                   "H_C":clean_acc2,"H_P":attacked_acc2, "H_D":defense_acc2,
                    "F_C":clean_acc3,"F_P":attacked_acc3, "F_D":defense_acc3 })

#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()




