# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

#(16.260162601626014, 0.0, 53.65853658536586, 0.0, 15.447154471544716, 0.0, 14.634146341463413) alpha 0 beta 1
#(16.304347826086957, 1.0869565217391304, 50.0, 0.0, 16.304347826086957, 0.0, 16.304347826086957) alpha 0.2 beta 0.8
#(18.181818181818183, 0.0, 45.45454545454545, 0.0, 12.121212121212121, 1.5151515151515151, 22.727272727272727) alpha 1 beta 0 
#(19.548872180451127, 0.0, 48.87218045112782, 0.0, 16.541353383458645, 0.0, 15.037593984962406) alpha 0.8 beta 0.2 
#(18.548387096774192, 0.0, 52.41935483870967, 0.0, 13.709677419354838, 0.0, 15.32258064516129) alpha 0.5 beta 0.5

species = ('n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n<not1&2hop>')
edge_distribution = {
    'propagation scheme 1': (16.260162601626014, 0.0, 53.65853658536586, 0.0, 15.447154471544716, 0.0, 14.634146341463413),
    'propagation scheme 2': (16.304347826086957, 1.0869565217391304, 50.0, 0.0, 16.304347826086957, 0.0, 16.304347826086957),

    'propagation scheme 3': (18.181818181818183, 0.0, 45.45454545454545, 0.0, 12.121212121212121, 1.5151515151515151, 22.727272727272727),
    'propagation scheme 4': (19.548872180451127, 0.0, 48.87218045112782, 0.0, 16.541353383458645, 0.0, 15.037593984962406),

    'propagation scheme 5': (18.548387096774192, 0.0, 52.41935483870967, 0.0, 13.709677419354838, 0.0, 15.32258064516129),
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



