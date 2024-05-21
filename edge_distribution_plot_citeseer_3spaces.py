# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np


#[18.181818181818183, 0.0, 38.38383838383838, 0.0, 14.14141414141414, 0.0, 29.292929292929294]
#[18.27956989247312, 0.0, 43.01075268817204, 0.0, 15.053763440860216, 0.0, 23.655913978494624]
#[12.643678160919542, 0.0, 39.08045977011494, 0.0, 12.643678160919542, 0.0, 35.63218390804598]


species = ('n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n<not1&2hop>')
edge_distribution = {
    'propagation space 1': (18.181818181818183, 0.0, 38.38383838383838, 0.0, 14.14141414141414, 0.0, 29.292929292929294),
    'propagation space 2': (18.27956989247312, 0.0, 43.01075268817204, 0.0, 15.053763440860216, 0.0, 23.655913978494624),
    'propagation space 3': (12.643678160919542, 0.0, 39.08045977011494, 0.0, 12.643678160919542, 0.0, 35.63218390804598),

}

x = np.arange(len(species))  # the label locations
width = 0.20  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in edge_distribution.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weight of Counts')
ax.set_title('Counts of Trapped Edges')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 80)

#plt.xlabel('Test Distribution')

plt.show()



