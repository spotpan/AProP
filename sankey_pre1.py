import csv
import numpy as np

def sort_and_label(numbers, k):
    # Sort the numbers in descending order and get the indices
    sorted_indices = np.argsort(numbers)[::-1]
    sorted_numbers = np.sort(numbers)[::-1]
    
    # Calculate the number of elements in each group
    group_size = len(sorted_numbers) // k
    
    # Assign labels to each element based on group
    labels = [i for i in range(k) for _ in range(group_size)]
    
    # Assign remaining labels to last group
    labels.extend([k-1] * (len(sorted_numbers) - len(labels)))
    
    # Map labels to the original order of numbers
    original_labels = [labels[sorted_indices[i]] for i in range(len(numbers))]
    
    return sorted_numbers, original_labels

# Function to map label index values to descriptions
def map_label_to_description(label_index):
    descriptions = ["strongest", "strong", "neuron", "weak", "weakest"]
    return descriptions[label_index]

# Function to create tuples for each index
def create_tuples(numbers, labels):
    tuples = []
    for i in range(len(numbers)):
        # Map label index to description
        description = map_label_to_description(labels[i])
        # Create tuple (index, sectX, label description, 1) where X is the section number
        tuples.append((i, f"sect{i+1}", description, 1.0))
    return tuples

# Example usage
numbers = [0.7105882352941176, 0.7129411764705882, 0.7364705882352941, 0.7035294117647058, 0.7176470588235293, 0.7035294117647058, 0.7247058823529411, 0.7435294117647058, 0.7317647058823529, 0.6941176470588235, 0.6941176470588235, 0.6823529411764705, 0.656470588235294, 0.6964705882352941, 0.7270588235294116, 0.6729411764705882, 0.708235294117647, 0.7058823529411764, 0.6823529411764705, 0.7035294117647058, 0.6894117647058823, 0.7105882352941176, 0.7247058823529411, 0.6847058823529412, 0.7058823529411764, 0.6964705882352941, 0.7129411764705882, 0.72, 0.7152941176470587, 0.7247058823529411, 0.6894117647058823, 0.6941176470588235, 0.7011764705882352, 0.6823529411764705, 0.7035294117647058, 0.7247058823529411, 0.6823529411764705]

k = 5  # Assuming there are 5 labels
sorted_numbers, labels = sort_and_label(numbers, k)
tuples = create_tuples(numbers, labels)

# Save tuples to a CSV file
filename = "labels.csv"
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "sect", "label description", "weight"])
    writer.writerows(tuples)

print("CSV file saved successfully.")
