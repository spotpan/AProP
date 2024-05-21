import torch
from hitsgnn import HITSGNN
from hitsgnn_rev2 import HITSGNN
#from gcn import GCN
from utils import *
import argparse
import numpy as np
from metattack import MetaApprox, Metattack
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from deeprobust.graph.data import Dataset, PrePtbDataset

import time


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

# === loading datasetss
#adj, features, labels = load_data(dataset=args.dataset)
data = Dataset(root='/tmp/', name='citeseer')
print(data)
adj, features, labels = data.adj, data.features, data.labels

nclass = max(labels) + 1

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)


print("idx_test", type(idx_test))


#idx_test=idx_test1

idx_unlabeled = np.union1d(idx_val, idx_test)




perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)



# set up attack model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=100, attack_features=False, lambda_=lambda_, device=device)

else:
    model = Metattack(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=100, attack_features=False, lambda_=lambda_, device=device)

if device != 'cpu':
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    model = model.to(device)


def test(adj):
    ''' test on HITSGNN '''

    adj = normalize_adj_tensor(adj)
    hitsgnn = HITSGNN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=0.5)

    if device != 'cpu':
        hitsgnn = hitsgnn.to(device)

    optimizer = optim.Adam(hitsgnn.parameters(),
                           lr=args.lr, weight_decay=5e-4)

    hitsgnn.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = hitsgnn(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

    hitsgnn.eval()
    output = hitsgnn(features, adj)


    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])


    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():

    
    #modified_adj = model(features, adj, labels, idx_train,
                         #idx_unlabeled, perturbations, ll_constraint=False)

    #modified_adj = modified_adj.detach()

    runs = 10
    clean_acc = []
    attacked_acc = []
    print('=== testing HITSGNN on original(clean) graph ===')

    start_time = time.time()

    for i in range(runs):
        #print(np.random.seed(args.seed))
        clean_acc.append(test(adj))

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("clean_acc:",clean_acc)

    print("Elapsed time", elapsed_time, "seconds")

    #print('=== testing HITSGNN on attacked graph ===')
    #for i in range(runs):
    #    attacked_acc.append(test(modified_adj))

    #print("attacked_acc",attacked_acc)



    #data=pd.DataFrame({"Acc. Clean":clean_acc,"Acc. Perturbed":attacked_acc})

    #plt.figure(figsize=(6,6))
    #sns.boxplot(data=data)#, re_trainings*[accuracy_logistic]])

    #plt.title("Accuracy before/after perturbing {}% edges using model {}".format(args.ptb_rate*100, args.model))
    #plt.savefig("results_on_{}.png".format(args.dataset), dpi=600)
    #plt.show()


if __name__ == '__main__':
    main()

#21.4ms

#0.1
#clean_acc: [0.7020142180094787, 0.7091232227488152, 0.7031990521327015, 0.6872037914691944, 0.705568720379147, 0.6925355450236967, 0.7037914691943129, 0.7008293838862559, 0.7026066350710901, 0.6996445497630333]

#0.2
#[0.7203791469194313, 0.7190250507786052, 0.7244414353419093, 0.7217332430602572, 0.7224102911306702, 0.7109004739336493, 0.7217332430602572, 0.7203791469194313, 0.7190250507786052, 0.7210561949898443]

#0.3
#[0.7266982622432859, 0.7274881516587678, 0.7259083728278041, 0.7259083728278041, 0.7203791469194313, 0.7306477093206951, 0.7266982622432859, 0.7282780410742496, 0.7322274881516587, 0.7251184834123222]

#0.4
#clean_acc: [0.7315689981096408, 0.724952741020794, 0.7334593572778828, 0.7296786389413988, 0.723062381852552, 0.727788279773157, 0.7344045368620038, 0.7296786389413988, 0.7334593572778828, 0.727788279773157]

#0.5
#clean_acc: [0.7520661157024793, 0.7473435655253837, 0.7355371900826446, 0.7520661157024793, 0.7473435655253837, 0.7473435655253837, 0.7638724911452184, 0.7520661157024793, 0.7544273907910272, 0.7390791027154663]

#0.6 seed=16
#clean_acc: [0.7767295597484277, 0.789308176100629, 0.7783018867924528, 0.7830188679245284, 0.779874213836478, 0.7845911949685535, 0.789308176100629, 0.7924528301886793, 0.779874213836478, 0.7767295597484277]

#0.6 seed=15 10
#clean_acc:[0.7704402515723271, 0.7814465408805031, 0.7908805031446541, 0.7688679245283019, 0.7704402515723271, 0.7688679245283019, 0.7657232704402516, 0.7720125786163522, 0.779874213836478, 0.7657232704402516]

#0.6 seed=15 20
#clean_acc: [0.7704402515723271, 0.7814465408805031, 0.7908805031446541, 0.7688679245283019, 0.7704402515723271, 0.7688679245283019, 0.7657232704402516, 0.7720125786163522, 0.779874213836478, 0.7657232704402516, 0.7720125786163522, 0.7578616352201258, 0.7767295597484277, 0.7641509433962265, 0.7767295597484277, 0.7783018867924528, 0.7688679245283019, 0.7735849056603774, 0.7751572327044025, 0.7657232704402516]

#0.6 seed=15 50
#[0.7704402515723271, 0.7814465408805031, 0.7908805031446541, 0.7688679245283019, 0.7704402515723271, 0.7688679245283019, 0.7657232704402516, 0.7720125786163522, 0.779874213836478, 0.7657232704402516, 0.7720125786163522, 0.7578616352201258, 0.7767295597484277, 0.7641509433962265, 0.7767295597484277, 0.7783018867924528, 0.7688679245283019, 0.7735849056603774, 0.7751572327044025, 0.7657232704402516, 0.7767295597484277, 0.7704402515723271, 0.7751572327044025, 0.7767295597484277, 0.7720125786163522, 0.7625786163522013, 0.7767295597484277, 0.779874213836478, 0.7672955974842768, 0.7751572327044025, 0.7751572327044025, 0.7735849056603774, 0.7720125786163522, 0.7783018867924528, 0.7751572327044025, 0.7704402515723271, 0.7688679245283019, 0.7688679245283019, 0.7720125786163522, 0.779874213836478, 0.7767295597484277, 0.7735849056603774, 0.7688679245283019, 0.7735849056603774, 0.7783018867924528, 0.7657232704402516, 0.7720125786163522, 0.7672955974842768, 0.7688679245283019, 0.7735849056603774]



#0.6 seed=17
#clean_acc: [0.7783018867924528, 0.7767295597484277, 0.7751572327044025, 0.7720125786163522, 0.7767295597484277, 0.7751572327044025, 0.7735849056603774, 0.7641509433962265, 0.7720125786163522, 0.779874213836478]

#0.6 seed=18
#clean_acc: [0.7374213836477987, 0.7327044025157233, 0.7484276729559749, 0.7279874213836478, 0.75, 0.7562893081761006, 0.7452830188679246, 0.7405660377358491, 0.7421383647798743, 0.7405660377358491]


#0.6 seed=19
#clean_acc: [0.7515723270440252, 0.7578616352201258, 0.7594339622641509, 0.7531446540880503, 0.7625786163522013, 0.7625786163522013, 0.7610062893081762, 0.7547169811320755, 0.7531446540880503, 0.7562893081761006]


#0.7 seed=15
#clean_acc: [0.776470588235294, 0.7741176470588235, 0.788235294117647, 0.776470588235294, 0.7788235294117647, 0.776470588235294, 0.7788235294117647, 0.7788235294117647, 0.7858823529411764, 0.776470588235294]

#0.7 seed=16
#clean_acc: [0.7670588235294117, 0.7717647058823529, 0.7670588235294117, 0.7717647058823529, 0.7576470588235293, 0.7694117647058822, 0.7647058823529411, 0.7670588235294117, 0.7670588235294117, 0.7552941176470588]

#0.7 seed=17
#clean_acc: [0.776470588235294, 0.7741176470588235, 0.7694117647058822, 0.7741176470588235, 0.7835294117647058, 0.7811764705882352, 0.7811764705882352, 0.7647058823529411, 0.7647058823529411, 0.7835294117647058]

#0.7 seed=18
#clean_acc: [0.7694117647058822, 0.7623529411764706, 0.7552941176470588, 0.7694117647058822, 0.7647058823529411, 0.7623529411764706, 0.7717647058823529, 0.7694117647058822, 0.7670588235294117, 0.7529411764705882]

#0.7 seed=19
#clean_acc: [0.7552941176470588, 0.7599999999999999, 0.7552941176470588, 0.7576470588235293, 0.7694117647058822, 0.7670588235294117, 0.7694117647058822, 0.7576470588235293, 0.7529411764705882, 0.7670588235294117]



#0.75+0.2+0.05 seed=19
#[0.7670588235294117, 0.7694117647058822, 0.7623529411764706, 0.7647058823529411, 0.7623529411764706, 0.7811764705882352, 0.7741176470588235, 0.7647058823529411, 0.7647058823529411, 0.7694117647058822]

#0.75+0.2+0.05 seed=18
#clean_acc: [0.7694117647058822, 0.7741176470588235, 0.7599999999999999, 0.7694117647058822, 0.7647058823529411, 0.7647058823529411, 0.7647058823529411, 0.7670588235294117, 0.7741176470588235, 0.7529411764705882]

#0.75+0.2+0.05 seed=17
#clean_acc: [0.7788235294117647, 0.7811764705882352, 0.7741176470588235, 0.776470588235294, 0.7858823529411764, 0.7858823529411764, 0.776470588235294, 0.7717647058823529, 0.7741176470588235, 0.7858823529411764]

#clean_acc: [0.776470588235294, 0.776470588235294, 0.7741176470588235, 0.7694117647058822, 0.7741176470588235, 0.7694117647058822, 0.7670588235294117, 0.7717647058823529, 0.7788235294117647, 0.776470588235294]




