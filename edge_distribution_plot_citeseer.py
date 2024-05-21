# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

#(17.582417582417584, 0.0, 39.56043956043956, 0.0, 14.285714285714285, 0.0, 28.57142857142857) alpha 0 beta 1
#(17.28395061728395, 0.0, 37.03703703703704, 0.0, 12.345679012345679, 0.0, 33.33333333333333) alpha 0.2 beta 0.8
#(23.655913978494624, 0.0, 36.55913978494624, 0.0, 13.978494623655912, 0.0, 25.806451612903224) alpha 1 beta 0 
#(18.085106382978726, 0.0, 44.680851063829785, 0.0, 13.829787234042554, 0.0, 23.404255319148938) alpha 0.8 beta 0.2 
#(12.643678160919542, 0.0, 39.08045977011494, 0.0, 12.643678160919542, 0.0, 35.63218390804598) alpha 0.5 beta 0.5

species = ('n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n<not1&2hop>')
edge_distribution = {
    'propagation scheme 1': (17.582417582417584, 0.0, 39.56043956043956, 0.0, 14.285714285714285, 0.0, 28.57142857142857),
    'propagation scheme 2': (17.28395061728395, 0.0, 37.03703703703704, 0.0, 12.345679012345679, 0.0, 33.33333333333333),

    'propagation scheme 3': (23.655913978494624, 0.0, 36.55913978494624, 0.0, 13.978494623655912, 0.0, 25.806451612903224),
    'propagation scheme 4': (18.085106382978726, 0.0, 44.680851063829785, 0.0, 13.829787234042554, 0.0, 23.404255319148938),

    'propagation scheme 5': (12.643678160919542, 0.0, 39.08045977011494, 0.0, 12.643678160919542, 0.0, 35.63218390804598),
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



