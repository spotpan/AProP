import pandas as pd
from pySankey.sankey import sankey

df = pd.read_csv(
    'pysankey/labels.csv', sep=',',
    names=['index', 'sect', 'label description', 'weight']
)

#index,sect,label description,weight

"""
rightcolor = {
    '#f71b1b',
    '#1b7ef7',
    '#f3f71b',
    '#12e23f',
    '#f78c1b'
}
"""

colorDict = {
    'strong':'#f71b1b',
    'strongest':'#1b7ef7',
    'neuron':'#f3f71b',
    'weak':'#12e23f',
    'weakest':'#f78c1b'
}

weight = df['weight'].values[1:].astype(float)

sankey(
    left=df['sect'].values[1:], right=df['label description'].values[1:], 
    rightWeight=1.6*weight, leftWeight=weight, 

    #rightColor=True,

    #colorDict=colorDict,

    aspect=20,
    fontsize=7, figure_name="labels",
    #curve_factor=0.3
)

# Result is in "fruit.png"


#[0.7105882352941176, 0.7129411764705882, 0.7364705882352941, 0.7035294117647058, 0.7176470588235293, 0.7035294117647058, 0.7247058823529411, 0.7435294117647058, 0.7317647058823529, 0.6941176470588235, 0.6941176470588235, 0.6823529411764705, 0.656470588235294, 0.6964705882352941, 0.7270588235294116, 0.6729411764705882, 0.708235294117647, 0.7058823529411764, 0.6823529411764705, 0.7035294117647058, 0.6894117647058823, 0.7105882352941176, 0.7247058823529411, 0.6847058823529412, 0.7058823529411764, 0.6964705882352941, 0.7129411764705882, 0.72, 0.7152941176470587, 0.7247058823529411, 0.6894117647058823, 0.6941176470588235, 0.7011764705882352, 0.6823529411764705, 0.7035294117647058, 0.7247058823529411, 0.6823529411764705]



