# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np


#[17.796610169491526, 0.0, 50.847457627118644, 0.0, 13.559322033898304, 0.0, 17.796610169491526]
#[18.6046511627907, 0.0, 55.81395348837209, 0.0, 13.953488372093023, 0.0, 11.627906976744185]
#[18.548387096774192, 0.0, 52.41935483870967, 0.0, 13.709677419354838, 0.0, 15.32258064516129]


species = ('n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n<not1&2hop>')
edge_distribution = {
    'propagation space 1': (17.796610169491526, 0.0, 50.847457627118644, 0.0, 13.559322033898304, 0.0, 17.796610169491526),
    'propagation space 2': (18.6046511627907, 0.0, 55.81395348837209, 0.0, 13.953488372093023, 0.0, 11.627906976744185),
    'propagation space 3': (18.548387096774192, 0.0, 52.41935483870967, 0.0, 13.709677419354838, 0.0, 15.32258064516129),

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



