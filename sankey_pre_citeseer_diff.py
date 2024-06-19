
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
            labels.append("strong")
        elif i < 2 * group_size:
            labels.append("neutral")
        #elif i < 3 * group_size:
        #    labels.append("neutral")
        #elif i < 4 * group_size:
        #    labels.append("weak")
        else:
            labels.append("weak")
    
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
tuples = [('n1', 'segment1', 0.047244094488189003), ('n1', 'segment2', 0.023622047244094446), ('n1', 'segment3', 0.03937007874015741), ('n1', 'segment4', 0.023622047244094446), ('n1', 'segment5', 0.011811023622047223), ('n1', 'segment6', 0.023622047244094446), ('n1', 'segment7', 0.03149606299212593), ('n1', 'segment8', 0.03937007874015752), ('n1', 'segment9', 0.03937007874015752), ('n1', 'segment10', 0.015748031496062964), ('n2', 'segment1', 0.010869565217391242), ('n2', 'segment2', 0.013586956521739135), ('n2', 'segment3', 0.0027173913043478937), ('n2', 'segment4', 0.013586956521739135), ('n2', 'segment5', 0.013586956521739135), ('n2', 'segment6', 0.013586956521739135), ('n2', 'segment7', 0.010869565217391353), ('n2', 'segment8', -0.005434782608695676), ('n2', 'segment9', 0.0190217391304347), ('n2', 'segment10', 0.005434782608695676), ('n3', 'segment1', 0.03863987635239574), ('n3', 'segment2', 0.057187017001545604), ('n3', 'segment3', 0.02782071097372496), ('n3', 'segment4', 0.06646058732612059), ('n3', 'segment5', 0.02782071097372496), ('n3', 'segment6', 0.04636785162287482), ('n3', 'segment7', 0.040185471406491424), ('n3', 'segment8', 0.010819165378670781), ('n3', 'segment9', 0.030911901081916437), ('n3', 'segment10', 0.026275115919629055), ('n4', 'segment1', -0.010152284263959421), ('n4', 'segment2', -0.040609137055837574), ('n4', 'segment3', 0.0), ('n4', 'segment4', 0.02030456852791873), ('n4', 'segment5', -0.005076142131979711), ('n4', 'segment6', -0.015228426395939132), ('n4', 'segment7', -0.015228426395939132), ('n4', 'segment8', -0.020304568527918843), ('n4', 'segment9', -0.025380710659898553), ('n4', 'segment10', -0.005076142131979711), ('n5', 'segment1', 0.03663003663003661), ('n5', 'segment2', 0.06593406593406592), ('n5', 'segment3', 0.05128205128205121), ('n5', 'segment4', 0.02564102564102555), ('n5', 'segment5', 0.06593406593406592), ('n5', 'segment6', 0.07326007326007322), ('n5', 'segment7', 0.05494505494505497), ('n5', 'segment8', 0.06227106227106216), ('n5', 'segment9', 0.04395604395604391), ('n5', 'segment10', 0.05128205128205121), ('n6', 'segment1', 0.012875536480686622), ('n6', 'segment2', -0.008583690987124415), ('n6', 'segment3', 0.0), ('n6', 'segment4', 0.0042918454935622075), ('n6', 'segment5', -0.008583690987124526), ('n6', 'segment6', 0.0), ('n6', 'segment7', 0.01716738197424894), ('n6', 'segment8', 0.008583690987124415), ('n6', 'segment9', -0.01716738197424883), ('n6', 'segment10', -0.008583690987124415)]

k = 3
sorted_tuples = sort_and_label_tuples(tuples, k)

# Save sorted tuples to CSV file
filename = "sorted_tuples_citeseer_diff.csv"
save_tuples_to_csv(sorted_tuples, filename)

print("CSV file saved successfully.")