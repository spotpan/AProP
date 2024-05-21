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


clean_acc1=[0.7327044025157233, 0.7374213836477987, 0.7248427672955975, 0.7279874213836478, 0.7327044025157233, 0.7264150943396227, 0.7279874213836478, 0.7327044025157233, 0.7279874213836478, 0.7421383647798743]
attacked_acc1=[0.6713836477987422, 0.6745283018867925, 0.6619496855345912, 0.6761006289308177, 0.6650943396226415, 0.669811320754717, 0.6713836477987422, 0.6808176100628931, 0.6713836477987422, 0.6713836477987422]
defense_acc1=[0.6666666666666667, 0.6745283018867925, 0.6729559748427674, 0.669811320754717, 0.669811320754717, 0.6745283018867925, 0.6682389937106918, 0.679245283018868, 0.669811320754717, 0.669811320754717]

clean_acc2=[0.7185534591194969, 0.7279874213836478, 0.7311320754716981, 0.7311320754716981, 0.738993710691824, 0.7374213836477987, 0.7342767295597484, 0.7232704402515724, 0.7358490566037736, 0.7405660377358491]
attacked_acc2=[0.6540880503144655, 0.6556603773584906, 0.6603773584905661, 0.6556603773584906, 0.6572327044025158, 0.6462264150943396, 0.6572327044025158, 0.6556603773584906, 0.6556603773584906, 0.6635220125786164]
defense_acc2=[0.6682389937106918, 0.6603773584905661, 0.6729559748427674, 0.6556603773584906, 0.6572327044025158, 0.6635220125786164, 0.6635220125786164, 0.6666666666666667, 0.6619496855345912, 0.6588050314465409]

clean_acc3=[0.7405660377358491, 0.7358490566037736, 0.7358490566037736, 0.7264150943396227, 0.729559748427673, 0.7279874213836478, 0.738993710691824, 0.7264150943396227, 0.729559748427673, 0.7342767295597484]
attacked_acc3= [0.7216981132075472, 0.7264150943396227, 0.7279874213836478, 0.7264150943396227, 0.7248427672955975, 0.7264150943396227, 0.7327044025157233, 0.7248427672955975, 0.7248427672955975, 0.7248427672955975]
defense_acc3=[0.7437106918238994, 0.7452830188679246, 0.7374213836477987, 0.738993710691824, 0.7468553459119497, 0.7468553459119497, 0.7484276729559749, 0.7437106918238994, 0.7405660377358491, 0.7468553459119497]

data=pd.DataFrame({"P_C":clean_acc1,"P_P":attacked_acc1, "P_D":defense_acc1, 
                   "H_C":clean_acc2,"H_P":attacked_acc2, "H_D":defense_acc2,
                    "F_C":clean_acc3,"F_P":attacked_acc3, "F_D":defense_acc3 })

#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()




