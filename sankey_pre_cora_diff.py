
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
tuples = [('n1', 'segment1', 0.036319612590798966), ('n1', 'segment2', 0.036319612590798966), ('n1', 'segment3', 0.014527845036319542), ('n1', 'segment4', 0.009685230024213065), ('n1', 'segment5', 0.02421307506053272), ('n1', 'segment6', 0.02421307506053272), ('n1', 'segment7', 0.014527845036319653), ('n1', 'segment8', 0.026634382566586012), ('n1', 'segment9', 0.053268765133171914), ('n1', 'segment10', 0.04358353510895885), ('n2', 'segment1', 0.053268765133171914), ('n2', 'segment2', 0.002421307506053294), ('n2', 'segment3', -0.002421307506053294), ('n2', 'segment4', 0.0), ('n2', 'segment5', 0.014527845036319653), ('n2', 'segment6', 0.03389830508474567), ('n2', 'segment7', 0.012106537530266248), ('n2', 'segment8', 0.002421307506053294), ('n2', 'segment9', 0.036319612590798966), ('n2', 'segment10', 0.01210653753026636), ('n3', 'segment1', 0.07118055555555558), ('n3', 'segment2', 0.0390625), ('n3', 'segment3', 0.06076388888888895), ('n3', 'segment4', 0.09201388888888895), ('n3', 'segment5', 0.05815972222222221), ('n3', 'segment6', 0.05208333333333337), ('n3', 'segment7', 0.04253472222222221), ('n3', 'segment8', 0.01822916666666663), ('n3', 'segment9', 0.06076388888888884), ('n3', 'segment10', 0.05902777777777779), ('n4', 'segment1', 0.026936026936026924), ('n4', 'segment2', 0.026936026936026924), ('n4', 'segment3', 0.01683501683501687), ('n4', 'segment4', 0.01683501683501676), ('n4', 'segment5', 0.0033670033670033517), ('n4', 'segment6', 0.006734006734006703), ('n4', 'segment7', 0.010101010101010055), ('n4', 'segment8', -0.010101010101010166), ('n4', 'segment9', -0.026936026936026924), ('n4', 'segment10', 0.023569023569023573), ('n5', 'segment1', 0.032835820895522394), ('n5', 'segment2', 0.06865671641791049), ('n5', 'segment3', 0.020895522388059695), ('n5', 'segment4', 0.05671641791044779), ('n5', 'segment5', 0.06567164179104479), ('n5', 'segment6', 0.011940298507462699), ('n5', 'segment7', 0.03880597014925369), ('n5', 'segment8', 0.023880597014925287), ('n5', 'segment9', 0.0507462686567165), ('n5', 'segment10', 0.032835820895522394), ('n6', 'segment1', -0.003154574132492205), ('n6', 'segment2', 0.01577287066246058), ('n6', 'segment3', -0.006309148264984188), ('n6', 'segment4', 0.018927444794952675), ('n6', 'segment5', 0.01577287066246058), ('n6', 'segment6', 0.012618296529968376), ('n6', 'segment7', -0.018927444794952675), ('n6', 'segment8', -0.009463722397476282), ('n6', 'segment9', -0.003154574132492094), ('n6', 'segment10', 0.006309148264984188)]


k = 3
sorted_tuples = sort_and_label_tuples(tuples, k)

# Save sorted tuples to CSV file
filename = "sorted_tuples_cora_diff.csv"
save_tuples_to_csv(sorted_tuples, filename)

print("CSV file saved successfully.")