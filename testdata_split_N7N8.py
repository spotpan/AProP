import torch
from gcn import GCN
from utils import *
import argparse
import numpy as np
from metattack import MetaApprox, Metattack
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx

# default=15
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

# === loading dataset
adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1

print("adj",adj.shape)
print("features",features.shape)
print("labels",labels.shape)


val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)

print("idx_train",len(idx_train))
#print("idx_val",idx_train)
#print("idx_test",idx_train)

idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

print("adj",adj)


################
##   type 1   ##
################

adj_matrix_tensor=adj
target_nodes_index=idx_train

# Assuming adj_matrix_tensor is the adjacency matrix tensor
# and target_nodes_index is the index of target nodes
adj_matrix = adj_matrix_tensor.numpy()

# Initialize a dictionary to store the indices of 2-hop neighbors for each target node
two_hop_neighbors_indices = []

# Iterate over each target node
for target_node_index in target_nodes_index:
    # Initialize a set to store unique indices of 2-hop neighbors
    two_hop_neighbors = set()
   
    # Get the row corresponding to the target node in the adjacency matrix
    row = adj_matrix[target_node_index]
   
    # Identify outgoing neighbors (type 1) and their neighbors (type 2)
    for neighbor_index, edge_weight in enumerate(row):
        if edge_weight != 0:  # Neighbor found
            # Add the neighbor itself (1-hop neighbor)
            # two_hop_neighbors.add(neighbor_index)
           
            # Add the neighbors of the neighbor (2-hop neighbors)
            neighbor_row = adj_matrix[neighbor_index]
            for neighbor_of_neighbor_index, neighbor_edge_weight in enumerate(neighbor_row):
                if neighbor_edge_weight != 0:  # Neighbor of neighbor found
                    two_hop_neighbors.add(neighbor_of_neighbor_index)
   
    # Add the set of indices of 2-hop neighbors to the dictionary
    two_hop_neighbors_indices.append(list(two_hop_neighbors))

#two_hop_neighbors_indices1=[ele for ele in two_hop_neighbors_indices if ele !=[]]
two_hop_neighbors_indices1=[ele for eleX in two_hop_neighbors_indices if eleX !=[] for ele in eleX]
two_hop_neighbors_indices1=list(set(two_hop_neighbors_indices1))
two_hop_neighbors_indices1=[ele for ele in two_hop_neighbors_indices1 if ele in idx_test]
print("len of 2_hop_neighbors_indices1",len(two_hop_neighbors_indices1))
# Now two_hop_neighbors_indices dictionary contains the indices of 2-hop neighbors for each target node
print("two_hop_neighbors_indices1",two_hop_neighbors_indices1)


################
##   type 2   ##
################



adj_matrix = adj_matrix_tensor.numpy()

two_hop_neighbors_indices = []

# Iterate over each target node
for target_node_index in target_nodes_index:
    # Initialize a set to store unique indices of 2-hop neighbors
    two_hop_neighbors = set()
   
    # Get the row corresponding to the target node in the adjacency matrix
    row = adj_matrix[target_node_index]
   
    # Identify outgoing neighbors (type 1) and their neighbors (type 2)
    for neighbor_index, edge_weight in enumerate(row):
        if edge_weight != 0:  # Neighbor found
            # Add the neighbor itself (1-hop neighbor)
            # two_hop_neighbors.add(neighbor_index)
           
            # Add the neighbors of the neighbor (2-hop neighbors)
            neighbor_col = adj_matrix[:,neighbor_index]
            for neighbor_of_neighbor_index, neighbor_edge_weight in enumerate(neighbor_col):
                if neighbor_edge_weight != 0:  # Neighbor of neighbor found
                    two_hop_neighbors.add(neighbor_of_neighbor_index)
   
    # Add the set of indices of 2-hop neighbors to the dictionary
    two_hop_neighbors_indices.append(list(two_hop_neighbors))

    #two_hop_neighbors_indices1=[ele for ele in two_hop_neighbors_indices if ele !=[]]
two_hop_neighbors_indices2=[ele for eleX in two_hop_neighbors_indices if eleX !=[] for ele in eleX]
two_hop_neighbors_indices2=list(set(two_hop_neighbors_indices2))
two_hop_neighbors_indices2=[ele for ele in two_hop_neighbors_indices2 if ele in idx_test]
print("len of 2_hop_neighbors_indices2",len(two_hop_neighbors_indices2))

print("two_hop_neighbors_indices2",two_hop_neighbors_indices2)


################
##   type 3   ##
################


adj_matrix = adj_matrix_tensor.numpy()

# Initialize a dictionary to store the indices of 2-hop neighbors for each target node
two_hop_neighbors_indices = []

# Iterate over each target node
for target_node_index in target_nodes_index:
    # Initialize a set to store unique indices of 2-hop neighbors
    two_hop_neighbors = set()
   
    # Get the row corresponding to the target node in the adjacency matrix
    col = adj_matrix[:,target_node_index]
   
    # Identify outgoing neighbors (type 1) and their neighbors (type 2)
    for neighbor_index, edge_weight in enumerate(col):
        if edge_weight != 0:  # Neighbor found
            # Add the neighbor itself (1-hop neighbor)
            # two_hop_neighbors.add(neighbor_index)
           
            # Add the neighbors of the neighbor (2-hop neighbors)
            neighbor_row = adj_matrix[neighbor_index]
            for neighbor_of_neighbor_index, neighbor_edge_weight in enumerate(neighbor_row):
                if neighbor_edge_weight != 0:  # Neighbor of neighbor found
                    two_hop_neighbors.add(neighbor_of_neighbor_index)
   
    # Add the set of indices of 2-hop neighbors to the dictionary
    two_hop_neighbors_indices.append(list(two_hop_neighbors))

#two_hop_neighbors_indices1=[ele for ele in two_hop_neighbors_indices if ele !=[]]
two_hop_neighbors_indices3=[ele for eleX in two_hop_neighbors_indices if eleX !=[] for ele in eleX]
two_hop_neighbors_indices3=list(set(two_hop_neighbors_indices3))
two_hop_neighbors_indices3=[ele for ele in two_hop_neighbors_indices3 if ele in idx_test]
print("len of 2_hop_neighbors_indices3",len(two_hop_neighbors_indices3))
# Now two_hop_neighbors_indices dictionary contains the indices of 2-hop neighbors for each target node
print("two_hop_neighbors_indices3",two_hop_neighbors_indices3)



################
##   type 4   ##
################


adj_matrix = adj_matrix_tensor.numpy()

# Initialize a dictionary to store the indices of 2-hop neighbors for each target node
two_hop_neighbors_indices = []

# Iterate over each target node
for target_node_index in target_nodes_index:
    # Initialize a set to store unique indices of 2-hop neighbors
    two_hop_neighbors = set()
   
    # Get the row corresponding to the target node in the adjacency matrix
    col = adj_matrix[:,target_node_index]
   
    # Identify outgoing neighbors (type 1) and their neighbors (type 2)
    for neighbor_index, edge_weight in enumerate(col):
        if edge_weight != 0:  # Neighbor found
            # Add the neighbor itself (1-hop neighbor)
            # two_hop_neighbors.add(neighbor_index)
           
            # Add the neighbors of the neighbor (2-hop neighbors)
            neighbor_col = adj_matrix[:,neighbor_index]
            for neighbor_of_neighbor_index, neighbor_edge_weight in enumerate(neighbor_col):
                if neighbor_edge_weight != 0:  # Neighbor of neighbor found
                    two_hop_neighbors.add(neighbor_of_neighbor_index)
   
    # Add the set of indices of 2-hop neighbors to the dictionary
    two_hop_neighbors_indices.append(list(two_hop_neighbors))

#two_hop_neighbors_indices1=[ele for ele in two_hop_neighbors_indices if ele !=[]]
two_hop_neighbors_indices4=[ele for eleX in two_hop_neighbors_indices if eleX !=[] for ele in eleX]
two_hop_neighbors_indices4=list(set(two_hop_neighbors_indices4))
two_hop_neighbors_indices4=[ele for ele in two_hop_neighbors_indices4 if ele in idx_test]
print("len of 2_hop_neighbors_indices4",len(two_hop_neighbors_indices4))
# Now two_hop_neighbors_indices dictionary contains the indices of 2-hop neighbors for each target node
print("two_hop_neighbors_indices4",two_hop_neighbors_indices4)




################
##   type 5   ##
################

adj_matrix_tensor=adj
target_nodes_index=idx_train

# Assuming adj_matrix_tensor is the adjacency matrix tensor
# and target_nodes_index is the index of target nodes
adj_matrix = adj_matrix_tensor.numpy()

# Initialize a dictionary to store the indices of 2-hop neighbors for each target node
one_hop_neighbors_indices = []

# Iterate over each target node
for target_node_index in target_nodes_index:
    # Initialize a set to store unique indices of 2-hop neighbors
    one_hop_neighbors = set()
   
    # Get the row corresponding to the target node in the adjacency matrix
    row = adj_matrix[target_node_index]
   
    # Identify outgoing neighbors (type 1) and their neighbors (type 2)
    for neighbor_index, edge_weight in enumerate(row):
        if edge_weight != 0:  # Neighbor found
            # Add the neighbor itself (1-hop neighbor)
            one_hop_neighbors.add(neighbor_index)
           
            # Add the neighbors of the neighbor (2-hop neighbors)
            #neighbor_row = adj_matrix[neighbor_index]
            #for neighbor_of_neighbor_index, neighbor_edge_weight in enumerate(neighbor_row):
                #if neighbor_edge_weight != 0:  # Neighbor of neighbor found
                    #two_hop_neighbors.add(neighbor_of_neighbor_index)
   
    # Add the set of indices of 1-hop neighbors to the dictionary
    one_hop_neighbors_indices.append(list(one_hop_neighbors))


one_hop_neighbors_indices5=[ele for eleX in one_hop_neighbors_indices if eleX !=[] for ele in eleX]
one_hop_neighbors_indices5=list(set(one_hop_neighbors_indices5))
one_hop_neighbors_indices5=[ele for ele in one_hop_neighbors_indices5 if ele in idx_test]
print("len of 1_hop_neighbors_indices5",len(one_hop_neighbors_indices5))
# Now one_hop_neighbors_indices dictionary contains the indices of 1-hop neighbors for each target node
print("one_hop_neighbors_indices5",one_hop_neighbors_indices5)






################
##   type 6   ##
################

adj_matrix_tensor=adj
target_nodes_index=idx_train

# Assuming adj_matrix_tensor is the adjacency matrix tensor
# and target_nodes_index is the index of target nodes
adj_matrix = adj_matrix_tensor.numpy()

# Initialize a dictionary to store the indices of 2-hop neighbors for each target node
one_hop_neighbors_indices = []

# Iterate over each target node
for target_node_index in target_nodes_index:
    # Initialize a set to store unique indices of 2-hop neighbors
    one_hop_neighbors = set()
   
    # Get the row corresponding to the target node in the adjacency matrix
    col = adj_matrix[:,target_node_index]
   
    # Identify outgoing neighbors (type 1) and their neighbors (type 2)
    for neighbor_index, edge_weight in enumerate(col):
        if edge_weight != 0:  # Neighbor found
            # Add the neighbor itself (1-hop neighbor)
            one_hop_neighbors.add(neighbor_index)
           
            # Add the neighbors of the neighbor (2-hop neighbors)
            #neighbor_row = adj_matrix[neighbor_index]
            #for neighbor_of_neighbor_index, neighbor_edge_weight in enumerate(neighbor_row):
                #if neighbor_edge_weight != 0:  # Neighbor of neighbor found
                    #two_hop_neighbors.add(neighbor_of_neighbor_index)
   
    # Add the set of indices of 1-hop neighbors to the dictionary
    one_hop_neighbors_indices.append(list(one_hop_neighbors))


one_hop_neighbors_indices6=[ele for eleX in one_hop_neighbors_indices if eleX !=[] for ele in eleX]
one_hop_neighbors_indices6=list(set(one_hop_neighbors_indices6))
one_hop_neighbors_indices6=[ele for ele in one_hop_neighbors_indices6 if ele in idx_test]
print("len of 1_hop_neighbors_indices6",len(one_hop_neighbors_indices6))
# Now one_hop_neighbors_indices dictionary contains the indices of 1-hop neighbors for each target node
print("one_hop_neighbors_indices6",one_hop_neighbors_indices6)




################
##   type 7   ##
################


print("len of 1_hop_neighbors_indices6",len(one_hop_neighbors_indices6))
one_hop_neighbors_indices7=[ele for ele in one_hop_neighbors_indices6 if ele not in two_hop_neighbors_indices1]
one_hop_neighbors_indices7=[ele for ele in one_hop_neighbors_indices7 if ele not in two_hop_neighbors_indices3]
print("len of 1_hop_neighbors_indices7",len(one_hop_neighbors_indices7))
# Now one_hop_neighbors_indices dictionary contains the indices of 1-hop neighbors for each target node
print("one_hop_neighbors_indices7",one_hop_neighbors_indices7)



################
##   type 8   ##
################


print("len of neighbors_indices8",len(two_hop_neighbors_indices4))
#neighbors_indices8=[ele for ele in two_hop_neighbors_indices4 if ele not in two_hop_neighbors_indices1]
neighbors_indices8=[ele for ele in two_hop_neighbors_indices4 if ele not in two_hop_neighbors_indices3]
print("neighbors_indices8",len(neighbors_indices8))
#neighbors_indices8=[ele for ele in neighbors_indices8 if ele not in one_hop_neighbors_indices5]
#print("neighbors_indices8",len(neighbors_indices8))
# Now one_hop_neighbors_indices dictionary contains the indices of 1-hop neighbors for each target node
print("neighbors_indices8",neighbors_indices8)


