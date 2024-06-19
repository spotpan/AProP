from pprint import pprint
from typing import Tuple, List
from IPython.display import display

import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go


import plotly.io as pio   
pio.kaleido.scope.mathjax = None


from itertools import product

states = {
    'Country': ['Canada', 'USA', 'Mexico'],
    'Occupation': ['Doctor', 'Lawyer', 'Banker'],
    'Life satisfaction': ['Happy', 'Meh', 'Unhappy']
}


rows = list(product(*states.values()))
vals = np.random.randint(100, size=len(rows))


df = pd.read_csv(
    'sorted_tuples_cora_diff.csv',  sep=',',usecols=[0, 1, 3]
    #names=['key1', 'key2', 'value', 'label']
)


#df_unique_paths = pd.DataFrame(rows, columns=states.keys()).assign(n=vals)

df_unique_paths = df.assign(n=1)


with pd.option_context('display.max_rows', 10):
    display(df_unique_paths)



def edges_from_unique_paths(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate a "long" set of edges, from a "wide" set of unique paths.

    Ignores any edges starting with an underscore.
    
    Args:
        df: DataFrame with n-1 columns that represent "levels",
            followed by a single numeric count/weight column.
        
    Returns:
        A DataFrame with three columns: (source, target, weight)
    
    """

    from collections import defaultdict
    edges = defaultdict(int)

    for idx, x in df.iterrows():
        only_visible = x.loc[lambda x: x.str.startswith('_') != True]
        n = only_visible['n']
        paths = only_visible.drop('n')
        for a, b in zip(paths[:-1], paths[1:]):
            edges[(a, b)] += n

    return pd.DataFrame(
        [(a, b, n) for (a, b), n in edges.items()],
        columns=['source', 'target', 'count']
    )

    
df_edges = edges_from_unique_paths(df_unique_paths)

with pd.option_context('display.max_rows', 10):
    display(df_edges)

from functools import reduce
    
def make_sankey_params_v1(df: pd.DataFrame) -> dict:
    """ Generate parameter dicts for go.Figure plotting function """
    
    # Unpack columns into lists
    sources, targets, values = df.values.T.tolist()
    
    # Create list of unique labels across node columns (source, target)
    labels = list(df['source'].pipe(set) | df['target'].pipe(set))
    
    # Map actual labels to their index value
    source_idx = list(map(labels.index, sources))
    target_idx = list(map(labels.index, targets))
    
    # Assemble final outputs into expected format
    nodes_dict = {'label': labels}
    links_dict = {
        'source': source_idx,
        'target': target_idx,
        'value': values
    }
    
    return nodes_dict, links_dict


nodes, links = make_sankey_params_v1(df_edges)


#pprint(nodes, compact=True)
#print('')
#pprint(links, compact=True)



data = go.Sankey(node=nodes, link=links)

#print("data", data)


#print("links",links)


mynodes = np.unique(df_edges[["source", "target"]], axis=None)
mynodes = pd.Series(index=mynodes, data=range(len(mynodes)))

#mynodes=mynodes[:]

print(mynodes)

"""
links.update({"color": [
                px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                for i in mynodes.loc[df_edges["target"]]
            ]})
"""
            
fig = go.Figure(data=[go.Sankey(
    node = nodes,
    link = links
  )])

default_margins = {'l': 25, 'r': 25, 't': 50, 'b': 50} 
#fig.update_layout(title="", margin=default_margins)

fig.update_layout(title_text="", font_size=7, width=300, height=300,margin=default_margins)

fig.write_image("pysankey/sankey_cora.pdf")

fig.show()



