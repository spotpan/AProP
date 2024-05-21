"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generating example data
np.random.seed(42)
data = pd.DataFrame({
    'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], size=100),
    'total_bill': np.random.normal(loc=20, scale=8, size=100)
})

# Create a figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the boxplot
sns.boxplot(x="day", y="total_bill", data=data, palette="husl")

# Get the unique values on x-axis and their corresponding positions
x_values = data['day'].unique()
x_positions = np.arange(len(x_values))

# Set the background color of the figure based on the variable on x-axis
for x_value, x_position in zip(x_values, x_positions):
    color = 'white' if x_value in ['Thur', 'Fri'] else (0.7,0.7,0.7)
    ax.axvspan(x_position - 0.5, x_position + 0.5, facecolor=color, alpha=0.2)

# Show the plot
plt.show()

"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example DataFrame
clean_acc1 = np.random.normal(loc=0.8, scale=0.1, size=100)
attacked_acc1 = np.random.normal(loc=0.75, scale=0.1, size=100)
clean_acc2 = np.random.normal(loc=0.7, scale=0.1, size=100)
attacked_acc2 = np.random.normal(loc=0.65, scale=0.1, size=100)
# Similarly for other variables

# Creating the DataFrame
data = pd.DataFrame({
    "C_0-1": clean_acc1,
    "P_0-1": attacked_acc1,
    "C_.2-.8": clean_acc2,
    "P_.2-.8": attacked_acc2,
    # Add other columns similarly
})

# Create a figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the boxplot
sns.boxplot(data=data, palette="husl", ax=ax)

# Get the unique values on x-axis and their corresponding positions
x_values = np.arange(len(data.columns))

# Set the background color of the figure based on the variable on x-axis
for x_position in x_values:
    col_name = data.columns[x_position]
    color = 'white' if col_name.startswith('C_') else (0.8, 0.8, 0.8)  # Light gray
    ax.axvspan(x_position - 0.5, x_position + 0.5, facecolor=color, alpha=0.5)

# Set x-axis labels
ax.set_xticks(x_values)
ax.set_xticklabels(data.columns, rotation=45, ha='right')


#plt.close('all')
# Show the plot
plt.tight_layout()
plt.show()

