
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
tuples = [('neighbor1', 'sect1', 0.8228346456692913), ('neighbor1', 'sect2', 0.7834645669291338), ('neighbor1', 'sect3', 0.7913385826771654), ('neighbor1', 'sect4', 0.7677165354330708), ('neighbor1', 'sect5', 0.7716535433070866), ('neighbor1', 'sect6', 0.8188976377952756), ('neighbor1', 'sect7', 0.7480314960629921), ('neighbor1', 'sect8', 0.7480314960629921), ('neighbor1', 'sect9', 0.7952755905511811), ('neighbor1', 'sect10', 0.7795275590551181), ('neighbor2', 'sect1', 0.7690217391304348), ('neighbor2', 'sect2', 0.7364130434782609), ('neighbor2', 'sect3', 0.7527173913043478), ('neighbor2', 'sect4', 0.7364130434782609), ('neighbor2', 'sect5', 0.6875), ('neighbor2', 'sect6', 0.7472826086956521), ('neighbor2', 'sect7', 0.7119565217391304), ('neighbor2', 'sect8', 0.6875), ('neighbor2', 'sect9', 0.7364130434782609), ('neighbor2', 'sect10', 0.7146739130434783), ('neighbor3', 'sect1', 0.7913446676970634), ('neighbor3', 'sect2', 0.7465224111282843), ('neighbor3', 'sect3', 0.7697063369397218), ('neighbor3', 'sect4', 0.7743431221020093), ('neighbor3', 'sect5', 0.7542503863987635), ('neighbor3', 'sect6', 0.7990726429675424), ('neighbor3', 'sect7', 0.7650695517774343), ('neighbor3', 'sect8', 0.7619783616692426), ('neighbor3', 'sect9', 0.7882534775888717), ('neighbor3', 'sect10', 0.7527047913446677), ('neighbor4', 'sect1', 0.7461928934010151), ('neighbor4', 'sect2', 0.715736040609137), ('neighbor4', 'sect3', 0.7411167512690354), ('neighbor4', 'sect4', 0.7360406091370558), ('neighbor4', 'sect5', 0.7106598984771573), ('neighbor4', 'sect6', 0.7766497461928933), ('neighbor4', 'sect7', 0.6954314720812182), ('neighbor4', 'sect8', 0.715736040609137), ('neighbor4', 'sect9', 0.7258883248730964), ('neighbor4', 'sect10', 0.7461928934010151), ('neighbor5', 'sect1', 0.8131868131868132), ('neighbor5', 'sect2', 0.7545787545787546), ('neighbor5', 'sect3', 0.7875457875457875), ('neighbor5', 'sect4', 0.7728937728937729), ('neighbor5', 'sect5', 0.7435897435897436), ('neighbor5', 'sect6', 0.8205128205128205), ('neighbor5', 'sect7', 0.7875457875457875), ('neighbor5', 'sect8', 0.7692307692307693), ('neighbor5', 'sect9', 0.7582417582417582), ('neighbor5', 'sect10', 0.7655677655677655), ('neighbor6', 'sect1', 0.7553648068669527), ('neighbor6', 'sect2', 0.7339055793991416), ('neighbor6', 'sect3', 0.6995708154506438), ('neighbor6', 'sect4', 0.7339055793991416), ('neighbor6', 'sect5', 0.6738197424892703), ('neighbor6', 'sect6', 0.7467811158798283), ('neighbor6', 'sect7', 0.6738197424892703), ('neighbor6', 'sect8', 0.6824034334763949), ('neighbor6', 'sect9', 0.7296137339055794), ('neighbor6', 'sect10', 0.7253218884120172)]



k = 5
sorted_tuples = sort_and_label_tuples(tuples, k)

# Save sorted tuples to CSV file
filename = "sorted_tuples_cora.csv"
save_tuples_to_csv(sorted_tuples, filename)

print("CSV file saved successfully.")