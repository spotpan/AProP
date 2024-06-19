
import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey( 
	node = dict( 
	thickness = 5, 
	line = dict(color = "green", width = 0.1), 
	label = ["A", "B", "C"], 
	color = "blue"
	), 
	link = dict( 
		
	# indices correspond to labels 
	source = [0, 6, 1], 
	target = [2, 1, 5], 
	value = [0, 1, 3] 
))]) 

fig.show()
