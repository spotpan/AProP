# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np


#[18.181818181818183, 0.0, 38.38383838383838, 0.0, 14.14141414141414, 0.0, 29.292929292929294]
#[18.27956989247312, 0.0, 43.01075268817204, 0.0, 15.053763440860216, 0.0, 23.655913978494624]
#[12.643678160919542, 0.0, 39.08045977011494, 0.0, 12.643678160919542, 0.0, 35.63218390804598]


species = ('PPNP', 'APPNP', 'N-GCN',  'GCN', 'GAT', 'HITS-GNN', 'HITS-GNN+', 'FPP')
edge_distribution = {
    'citeseer': (18.9, 23.1, 115.9, 35.3, 187.0, 17.3, 6.0, 21.4),
    'cora-ML': (29.7, 36.2, 119.8, 36.5, 217.4, 20.1, 8.71, 28.9),
    'pubmed': (0, 43.3, 342.6, 48.3, 1029.8, 72.3, 39.6, 220.0),

}

x = np.arange(len(species))  # the label locations
width = 0.20  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

color=['lightblue', 'grey', 'lightcoral']

for attribute, measurement in edge_distribution.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=color[multiplier])
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Training time per epoch (ms)')
#ax.set_title('Counts of Trapped Edges')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 1200)

#plt.xlabel('Test Distribution')

plt.show()



