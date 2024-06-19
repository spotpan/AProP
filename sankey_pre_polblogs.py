
import csv
import numpy as np

def sort_and_label_tuples(tuples, k):
    # Extract the values from tuples
    values = [t[2] for t in tuples]
    
    # Sort the values in descending order and get the indices
    sorted_indices = np.argsort(values)[::-1]

    print("indices",sorted_indices)

    sorted_values = np.sort(values)[::-1]

    print("values",sorted_values)
    
    # Calculate the number of elements in each group
    group_size = len(sorted_values) // k
    
    # Assign labels to each element based on group
    labels = []
    for i, val in enumerate(sorted_values):
        if i < group_size:
            labels.append("strongest")
        elif i < 2 * group_size:
            labels.append("strong")
        elif i < 3 * group_size:
            labels.append("neutral")
        elif i < 4 * group_size:
            labels.append("weak")
        else:
            labels.append("weakest")
    
    print("labels",labels)
    # Combine original labels with tuples
    tuples = [(tuples[i][0], tuples[i][1], tuples[i][2]) for i in sorted_indices]

    sorted_tuples = [(tuples[i][0], tuples[i][1], tuples[i][2], labels[i]) for i in range(len(tuples))]
    
    #sorted_tuples = [(tuples[i][0], tuples[i][1], tuples[i][2], tuples[i][3]) for i in sorted_indices]


    return sorted_tuples

# Function to save tuples to CSV file
def save_tuples_to_csv(tuples, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key1", "key2", "value", "label"])
        writer.writerows(tuples)

# Example usage
tuples =  [('neighbor1', 'sect1', 0.9624819624819625), ('neighbor1', 'sect2', 0.9595959595959596), ('neighbor1', 'sect3', 0.9668109668109668), ('neighbor1', 'sect4', 0.9624819624819625), ('neighbor1', 'sect5', 0.963924963924964), ('neighbor1', 'sect6', 0.9567099567099567), ('neighbor1', 'sect7', 0.9191919191919192), ('neighbor1', 'sect8', 0.9148629148629148), ('neighbor1', 'sect9', 0.9307359307359307), ('neighbor1', 'sect10', 0.9134199134199135), ('neighbor2', 'sect1', 0.9608540925266904), ('neighbor2', 'sect2', 0.9572953736654805), ('neighbor2', 'sect3', 0.9596678529062871), ('neighbor2', 'sect4', 0.9596678529062871), ('neighbor2', 'sect5', 0.9572953736654805), ('neighbor2', 'sect6', 0.9561091340450771), ('neighbor2', 'sect7', 0.9478054567022539), ('neighbor2', 'sect8', 0.9478054567022539), ('neighbor2', 'sect9', 0.937129300118624), ('neighbor2', 'sect10', 0.9395017793594307), ('neighbor3', 'sect1', 0.9565217391304348), ('neighbor3', 'sect2', 0.9617918313570488), ('neighbor3', 'sect3', 0.9578392621870884), ('neighbor3', 'sect4', 0.9499341238471674), ('neighbor3', 'sect5', 0.9525691699604744), ('neighbor3', 'sect6', 0.9459815546772069), ('neighbor3', 'sect7', 0.9117259552042161), ('neighbor3', 'sect8', 0.9183135704874835), ('neighbor3', 'sect9', 0.927536231884058), ('neighbor3', 'sect10', 0.9025032938076417), ('neighbor4', 'sect1', 0.9635678391959799), ('neighbor4', 'sect2', 0.9597989949748744), ('neighbor4', 'sect3', 0.9623115577889447), ('neighbor4', 'sect4', 0.9660804020100503), ('neighbor4', 'sect5', 0.9610552763819096), ('neighbor4', 'sect6', 0.9635678391959799), ('neighbor4', 'sect7', 0.9522613065326633), ('neighbor4', 'sect8', 0.9384422110552764), ('neighbor4', 'sect9', 0.9472361809045227), ('neighbor4', 'sect10', 0.9472361809045227), ('neighbor5', 'sect1', 0.9764397905759163), ('neighbor5', 'sect2', 0.9712041884816754), ('neighbor5', 'sect3', 0.9764397905759163), ('neighbor5', 'sect4', 0.9738219895287958), ('neighbor5', 'sect5', 0.9764397905759163), ('neighbor5', 'sect6', 0.9659685863874347), ('neighbor5', 'sect7', 0.9659685863874347), ('neighbor5', 'sect8', 0.9607329842931938), ('neighbor5', 'sect9', 0.9659685863874347), ('neighbor5', 'sect10', 0.9633507853403142), ('neighbor6', 'sect1', 0.9668989547038328), ('neighbor6', 'sect2', 0.9651567944250871), ('neighbor6', 'sect3', 0.9738675958188153), ('neighbor6', 'sect4', 0.9564459930313589), ('neighbor6', 'sect5', 0.9686411149825784), ('neighbor6', 'sect6', 0.9634146341463414), ('neighbor6', 'sect7', 0.9616724738675958), ('neighbor6', 'sect8', 0.9512195121951219), ('neighbor6', 'sect9', 0.9320557491289199), ('neighbor6', 'sect10', 0.9547038327526133)]



k = 5
sorted_tuples = sort_and_label_tuples(tuples, k)

# Save sorted tuples to CSV file
filename = "sorted_tuples_polblogs.csv"
save_tuples_to_csv(sorted_tuples, filename)

print("CSV file saved successfully.")