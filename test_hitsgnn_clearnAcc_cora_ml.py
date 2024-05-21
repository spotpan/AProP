import torch
#from hitsgnn import HITSGNN
from hitsgnn_rev2 import HITSGNN
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
data = Dataset(root='/tmp/', name='cora_ml')
print(data)
adj, features, labels = data.adj, data.features, data.labels

nclass = max(labels) + 1

val_size = 0.1
test_size = 0.401
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

    runs = 50
    clean_acc = []
    attacked_acc = []
    print('=== testing HITSGNN on original(clean) graph ===')

    start_time = time.time()

    for i in range(runs):
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

#28.9ms 

#0.1
#clean_acc: [0.8309608540925266, 0.8403024911032028, 0.8327402135231317, 0.8456405693950177, 0.8358540925266903, 0.842526690391459, 0.8416370106761566, 0.8434163701067615, 0.8385231316725978, 0.842526690391459]

#0.2
#clean_acc: [0.8439247585155059, 0.8418912048805287, 0.8449415353329944, 0.8439247585155059, 0.8439247585155059, 0.8413828164717845, 0.8510421962379258, 0.8403660396542959, 0.8469750889679716, 0.842399593289273]

#0.3
#clean_acc: [0.8588374851720048, 0.8588374851720048, 0.8582443653618032, 0.8594306049822065, 0.8576512455516014, 0.8588374851720048, 0.8629893238434164, 0.8600237247924081, 0.8600237247924081, 0.8552787663107948]

#0.4
#clean_acc: [0.8562277580071175, 0.8569395017793595, 0.8512455516014236, 0.8533807829181496, 0.8562277580071175, 0.8548042704626335, 0.8562277580071175, 0.8526690391459075, 0.8491103202846976, 0.8569395017793595]

#0.5 seed=15 10
#clean_acc: [0.8624667258207631, 0.8615794143744454, 0.8660159716060337, 0.8642413487133984, 0.8633540372670808, 0.8615794143744454, 0.8651286601597161, 0.8562555456965395, 0.8606921029281278, 0.8642413487133984]

#0.5 seed=15 50
#[0.8624667258207631, 0.8615794143744454, 0.8660159716060337, 0.8642413487133984, 0.8633540372670808, 0.8615794143744454, 0.8651286601597161, 0.8562555456965395, 0.8606921029281278, 0.8642413487133984, 0.8615794143744454, 0.8589174800354925, 0.8642413487133984, 0.8589174800354925, 0.8660159716060337, 0.8580301685891748, 0.8589174800354925, 0.8544809228039042, 0.8651286601597161, 0.8562555456965395, 0.8606921029281278, 0.8642413487133984, 0.8642413487133984, 0.8562555456965395, 0.8624667258207631, 0.8598047914818101, 0.8615794143744454, 0.8642413487133984, 0.8589174800354925, 0.8624667258207631, 0.8633540372670808, 0.8624667258207631, 0.8606921029281278, 0.867790594498669, 0.8633540372670808, 0.8624667258207631, 0.8624667258207631, 0.8598047914818101, 0.8624667258207631, 0.8615794143744454, 0.8544809228039042, 0.867790594498669, 0.8615794143744454, 0.8580301685891748, 0.8580301685891748, 0.8624667258207631, 0.8624667258207631, 0.8571428571428572, 0.8589174800354925, 0.8615794143744454]



#0.6
#clean_acc: [0.8581560283687943, 0.8581560283687943, 0.8605200945626478, 0.8605200945626478, 0.8569739952718676, 0.8640661938534279, 0.8605200945626478, 0.8557919621749409, 0.8628841607565012, 0.8522458628841607]


