
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
tuples = [('neighbor1', 'sect1', 0.7007874015748031), ('neighbor1', 'sect2', 0.7283464566929134), ('neighbor1', 'sect3', 0.7401574803149606), ('neighbor1', 'sect4', 0.7086614173228346), ('neighbor1', 'sect5', 0.7125984251968503), ('neighbor1', 'sect6', 0.7086614173228346), ('neighbor1', 'sect7', 0.7047244094488189), ('neighbor1', 'sect8', 0.7086614173228346), ('neighbor1', 'sect9', 0.7165354330708661), ('neighbor1', 'sect10', 0.7047244094488189), ('neighbor2', 'sect1', 0.6630434782608695), ('neighbor2', 'sect2', 0.6684782608695652), ('neighbor2', 'sect3', 0.6766304347826086), ('neighbor2', 'sect4', 0.6793478260869565), ('neighbor2', 'sect5', 0.6766304347826086), ('neighbor2', 'sect6', 0.6739130434782609), ('neighbor2', 'sect7', 0.6793478260869565), ('neighbor2', 'sect8', 0.6875), ('neighbor2', 'sect9', 0.6603260869565217), ('neighbor2', 'sect10', 0.6739130434782609), ('neighbor3', 'sect1', 0.7341576506955177), ('neighbor3', 'sect2', 0.7372488408037094), ('neighbor3', 'sect3', 0.732612055641422), ('neighbor3', 'sect4', 0.731066460587326), ('neighbor3', 'sect5', 0.732612055641422), ('neighbor3', 'sect6', 0.7264296754250387), ('neighbor3', 'sect7', 0.7295208655332303), ('neighbor3', 'sect8', 0.732612055641422), ('neighbor3', 'sect9', 0.714064914992272), ('neighbor3', 'sect10', 0.7217928902627512), ('neighbor4', 'sect1', 0.700507614213198), ('neighbor4', 'sect2', 0.7106598984771573), ('neighbor4', 'sect3', 0.7106598984771573), ('neighbor4', 'sect4', 0.6903553299492385), ('neighbor4', 'sect5', 0.6954314720812182), ('neighbor4', 'sect6', 0.7208121827411167), ('neighbor4', 'sect7', 0.715736040609137), ('neighbor4', 'sect8', 0.715736040609137), ('neighbor4', 'sect9', 0.7055837563451776), ('neighbor4', 'sect10', 0.6954314720812182), ('neighbor5', 'sect1', 0.6959706959706959), ('neighbor5', 'sect2', 0.7032967032967032), ('neighbor5', 'sect3', 0.6959706959706959), ('neighbor5', 'sect4', 0.7032967032967032), ('neighbor5', 'sect5', 0.6996336996336996), ('neighbor5', 'sect6', 0.6996336996336996), ('neighbor5', 'sect7', 0.6813186813186813), ('neighbor5', 'sect8', 0.7032967032967032), ('neighbor5', 'sect9', 0.6923076923076923), ('neighbor5', 'sect10', 0.6996336996336996), ('neighbor6', 'sect1', 0.721030042918455), ('neighbor6', 'sect2', 0.7296137339055794), ('neighbor6', 'sect3', 0.7296137339055794), ('neighbor6', 'sect4', 0.7296137339055794), ('neighbor6', 'sect5', 0.7253218884120172), ('neighbor6', 'sect6', 0.7296137339055794), ('neighbor6', 'sect7', 0.7167381974248928), ('neighbor6', 'sect8', 0.7339055793991416), ('neighbor6', 'sect9', 0.721030042918455), ('neighbor6', 'sect10', 0.7296137339055794)]


k = 5
sorted_tuples = sort_and_label_tuples(tuples, k)

# Save sorted tuples to CSV file
filename = "sorted_tuples_citeseer.csv"
save_tuples_to_csv(sorted_tuples, filename)

print("CSV file saved successfully.")