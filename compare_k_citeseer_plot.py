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


clean_acc1=[0.738993710691824, 0.729559748427673, 0.7169811320754718, 0.7279874213836478, 0.7279874213836478, 0.7358490566037736, 0.7279874213836478, 0.7311320754716981, 0.738993710691824, 0.7405660377358491]
attacked_acc1=[0.6635220125786164, 0.6682389937106918, 0.6650943396226415, 0.6572327044025158, 0.6588050314465409, 0.6650943396226415, 0.6682389937106918, 0.6666666666666667, 0.6603773584905661, 0.6650943396226415]

clean_acc2=[0.7547169811320755, 0.7468553459119497, 0.7452830188679246, 0.7531446540880503, 0.7641509433962265, 0.7484276729559749, 0.7421383647798743, 0.7437106918238994, 0.7374213836477987, 0.7452830188679246]
attacked_acc2=[0.7248427672955975, 0.7468553459119497, 0.7374213836477987, 0.7311320754716981, 0.7327044025157233, 0.738993710691824, 0.7327044025157233, 0.7374213836477987, 0.7374213836477987, 0.7437106918238994]

clean_acc3=[0.7405660377358491, 0.7374213836477987, 0.7610062893081762, 0.738993710691824, 0.7421383647798743, 0.7578616352201258, 0.6761006289308177, 0.75, 0.7185534591194969, 0.738993710691824]
attacked_acc3= [0.7374213836477987, 0.7405660377358491, 0.7358490566037736, 0.7358490566037736, 0.7594339622641509, 0.7311320754716981, 0.7075471698113208, 0.7342767295597484, 0.7437106918238994, 0.738993710691824]

clean_acc4=[0.5770440251572327, 0.5172955974842768, 0.25157232704402516, 0.5534591194968553, 0.25157232704402516, 0.4889937106918239, 0.30188679245283023, 0.550314465408805, 0.47327044025157233, 0.25157232704402516]
attacked_acc4= [0.529874213836478, 0.5110062893081762, 0.369496855345912, 0.25157232704402516, 0.25157232704402516, 0.25157232704402516, 0.25157232704402516, 0.42138364779874216, 0.2531446540880503, 0.529874213836478]


data=pd.DataFrame({"C_k=2":clean_acc1,"P_k=2":attacked_acc1, 
                   "C_k=6":clean_acc2,"P_k=6":attacked_acc2, 
                    "C_k=12":clean_acc3,"P_k=12":attacked_acc3,
                    "C_k=24":clean_acc4,"P_k=24":attacked_acc4 })

#plt.figure(figsize=(6,6))
sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

plt.title("Accuracy before/after perturbing {}% edges".format(args.ptb_rate*100, args.model))
plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
plt.show()




