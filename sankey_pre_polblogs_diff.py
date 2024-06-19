
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
tuples =  [('n1', 'segment1', 0.06493506493506496), ('n1', 'segment2', 0.07647907647907648), ('n1', 'segment3', 0.06349206349206349), ('n1', 'segment4', 0.053391053391053434), ('n1', 'segment5', 0.08369408369408371), ('n1', 'segment6', 0.06349206349206349), ('n1', 'segment7', 0.0548340548340549), ('n1', 'segment8', 0.03463203463203457), ('n1', 'segment9', 0.044733044733044736), ('n1', 'segment10', 0.037518037518037506), ('n2', 'segment1', 0.010676156583629859), ('n2', 'segment2', 0.008303677342823224), ('n2', 'segment3', 0.011862396204033177), ('n2', 'segment4', 0.005931198102016588), ('n2', 'segment5', 0.004744958481613271), ('n2', 'segment6', 0.008303677342823224), ('n2', 'segment7', 0.027283511269276417), ('n2', 'segment8', 0.004744958481613271), ('n2', 'segment9', 0.0023724792408066353), ('n2', 'segment10', -0.013048635824436494), ('n3', 'segment1', 0.08168642951251648), ('n3', 'segment2', 0.08695652173913049), ('n3', 'segment3', 0.07509881422924902), ('n3', 'segment4', 0.06851119894598157), ('n3', 'segment5', 0.08959156785243738), ('n3', 'segment6', 0.06587615283267456), ('n3', 'segment7', 0.04216073781291163), ('n3', 'segment8', 0.03820816864295129), ('n3', 'segment9', 0.0487483530961792), ('n3', 'segment10', 0.022397891963109373), ('n4', 'segment1', 0.017587939698492483), ('n4', 'segment2', 0.008793969849246186), ('n4', 'segment3', 0.0037688442211055717), ('n4', 'segment4', -0.005025125628140614), ('n4', 'segment5', 0.008793969849246186), ('n4', 'segment6', 0.007537688442211032), ('n4', 'segment7', 0.012562814070351758), ('n4', 'segment8', 0.002512562814070418), ('n4', 'segment9', 0.011306532663316604), ('n4', 'segment10', 0.013819095477386911), ('n5', 'segment1', 0.020942408376963373), ('n5', 'segment2', 0.03403141361256545), ('n5', 'segment3', 0.01832460732984298), ('n5', 'segment4', 0.010471204188481686), ('n5', 'segment5', 0.020942408376963373), ('n5', 'segment6', 0.015706806282722474), ('n5', 'segment7', 0.023560209424083767), ('n5', 'segment8', 0.01308900523560208), ('n5', 'segment9', 0.023560209424083767), ('n5', 'segment10', 0.018324607329842868), ('n6', 'segment1', 0.012195121951219523), ('n6', 'segment2', -0.0017421602787456303), ('n6', 'segment3', 0.013937282229965153), ('n6', 'segment4', 0.0017421602787457413), ('n6', 'segment5', 0.008710801393728262), ('n6', 'segment6', 0.010452961672473893), ('n6', 'segment7', 0.022648083623693305), ('n6', 'segment8', 0.008710801393728262), ('n6', 'segment9', 0.0034843205574912606), ('n6', 'segment10', 0.0034843205574913716)]


k = 3
sorted_tuples = sort_and_label_tuples(tuples, k)

# Save sorted tuples to CSV file
filename = "sorted_tuples_polblogs_diff.csv"
save_tuples_to_csv(sorted_tuples, filename)

print("CSV file saved successfully.")
