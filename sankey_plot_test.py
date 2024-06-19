import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import itertools

df = pd.DataFrame(
    itertools.product(
        ["AF", "SY", "IQ", "SO", "ER"], ["DE", "AT", "BG", "SE", "UK", "CH"]
    ),
    columns=["source", "target"],
).pipe(lambda d: d.assign(value=np.random.uniform(1, 10000, 1000)[:len(d)]))

nodes = np.unique(df[["source", "target"]], axis=None)
nodes = pd.Series(index=nodes, data=range(len(nodes)))

print("nodes",nodes)

fig = go.Figure(
    go.Sankey(
        node={
            "label": nodes.index,
            "color": [
                px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                for i in nodes
            ],
        },
        link={
            "source": nodes.loc[df["source"]],
            "target": nodes.loc[df["target"]],
            "value": df["value"],
            "color": [
                px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                for i in nodes.loc[df["source"]]
            ],
        },
    )
)

#fig.show()
