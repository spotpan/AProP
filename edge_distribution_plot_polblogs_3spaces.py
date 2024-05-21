# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np


#[14.661654135338345, 19.172932330827066, 16.541353383458645, 16.541353383458645, 8.270676691729323, 10.150375939849624, 14.661654135338345]
#[13.385826771653544, 20.078740157480315, 14.960629921259844, 17.322834645669293, 7.480314960629922, 11.023622047244094, 15.748031496062993]
#[15.579710144927535, 19.565217391304348, 17.02898550724638, 17.02898550724638, 8.695652173913043, 9.057971014492754, 13.043478260869565]

species = ('n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n<not1&2hop>')
edge_distribution = {
    'propagation space 1': (14.661654135338345, 19.172932330827066, 16.541353383458645, 16.541353383458645, 8.270676691729323, 10.150375939849624, 14.661654135338345),
    'propagation space 2': (13.385826771653544, 20.078740157480315, 14.960629921259844, 17.322834645669293, 7.480314960629922, 11.023622047244094, 15.748031496062993),
    'propagation space 3': (15.579710144927535, 19.565217391304348, 17.02898550724638, 17.02898550724638, 8.695652173913043, 9.057971014492754, 13.043478260869565),

}

x = np.arange(len(species))  # the label locations
width = 0.15  # the width of the bars
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



