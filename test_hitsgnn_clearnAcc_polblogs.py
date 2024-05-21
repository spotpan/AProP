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
data = Dataset(root='/tmp/', name='polblogs')
print(data)
adj, features, labels = data.adj, data.features, data.labels


print("features",features)

nclass = max(labels) + 1

val_size = 0.1
test_size = 0.3
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
    for i in range(runs):
        clean_acc.append(test(adj))

    print("clean_acc:",clean_acc)

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


#pubmed 0.1
#clean_acc: [0.8299733739064283, 0.8301635602890833, 0.8294028147584633, 0.8304171421326234, 0.8317484468112084, 0.8310510967414734, 0.8297197920628884, 0.8310510967414734, 0.8294662102193484, 0.8292126283758083]

#0.2 
#clean_acc: [0.8339371105636865, 0.8321257788726272, 0.8357484422547458, 0.8338646572960441, 0.835603535719461, 0.8334299376901899, 0.8318359658020578, 0.834444283437183, 0.8332125778872628, 0.8322706854079119]

#0.3 seed=15 10
#clean_acc: [0.8324740089595132, 0.8321359141239117, 0.8332347223396163, 0.8318823429972108, 0.8321359141239117, 0.8321359141239117, 0.8310371059082072, 0.8315442481616093, 0.8317132955794101, 0.8311216296171077]


#0.6 seed=15 50
#clean_acc: [0.8345165652467884, 0.8378972278566599, 0.837052062204192, 0.8340094658553077, 0.8355307640297498, 0.834685598377282, 0.8343475321162948, 0.8353617308992562, 0.8356997971602433, 0.8343475321162948, 0.8343475321162948, 0.834685598377282, 0.8353617308992562, 0.834685598377282, 0.8329952670723462, 0.834685598377282, 0.834685598377282, 0.835868830290737, 0.835023664638269, 0.8387423935091278, 0.834685598377282, 0.833840432724814, 0.8345165652467884, 0.8341784989858012, 0.8365449628127113, 0.834685598377282, 0.8336713995943205, 0.835868830290737, 0.8355307640297498, 0.833840432724814, 0.8321501014198783, 0.8356997971602433, 0.8341784989858012, 0.8343475321162948, 0.835868830290737, 0.8345165652467884, 0.8348546315077755, 0.8314739688979039, 0.8353617308992562, 0.8356997971602433, 0.835023664638269, 0.834685598377282, 0.8348546315077755, 0.8348546315077755, 0.8351926977687627, 0.8328262339418526, 0.8333333333333334, 0.8331643002028397, 0.8329952670723462, 0.835868830290737]

