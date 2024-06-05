import pandas as pd
import matplotlib.pyplot as plt

def nan_count(file_path, column_name):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Create histogram
    nan_counts = data.groupby([data['firstAlgorithmName'], data['secondAlgorithmName']])[column_name].apply(lambda x: x.isna().sum())
    nan_counts.plot(kind='bar', figsize=(10, 6), title='NaN Count in Column: {}'.format(column_name))
    plt.xlabel('Algorithm Names')
    plt.ylabel('Injection count where: '+column_name+' is Nan')
    plt.xticks(rotation=90)
    plt.show()

def nan_on_bitpos(injection_file, fault_file, column_name):
    nan_count_utils(injection_file, fault_file, 'bit_pos', column_name, False)

def nan_count_utils(injection_file, fault_file, column_groupby, column_shown, filter):
    # Read CSV files
    injection_data = pd.read_csv(injection_file)
    fault_data = pd.read_csv(fault_file)

    # Merge the two tables based on the common column 'injectionId'
    merged_data = pd.merge(injection_data, fault_data, left_on='injectionId', right_on='inj_id')

    # Count NaN occurrences in the specified column for each group
    nan_counts = merged_data.groupby([column_groupby])[column_shown].apply(lambda x: x.isna().sum())

    if(filter):
        # Filter nan counts strictly more than 0
        nan_counts = nan_counts[nan_counts > 0]

    if not nan_counts.empty:
        # Plot the graph
        nan_counts.plot(kind='bar', figsize=(10, 6), title='NaN Occurrences in Column: {}'.format(column_shown))
        plt.xlabel(column_groupby)
        plt.ylabel('Injection count where: '+column_groupby+' is Nan')
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("No groups with NaN occurrences in column '{}'".format(column_shown))

def nan_tensor_inside(injection_file, fault_file, column_shown):
    # Read CSV files
    injection_data = pd.read_csv(injection_file)
    fault_data = pd.read_csv(fault_file)

    # Merge the two tables based on the common column 'injectionId'
    merged_data = pd.merge(injection_data, fault_data, left_on='injectionId', right_on='inj_id')

    # Count NaN occurrences in the specified column for each group
    nan_counts = merged_data.groupby(['injectionId','n','c','h','w'])[column_shown].apply(lambda x: x.isna().sum())

    # Filter nan counts strictly more than 0
    nan_counts = nan_counts[nan_counts > 0]

    if not nan_counts.empty:
        # Plot the graph
        nan_counts.plot(kind='bar', figsize=(10, 6), title='NaN Occurrences in Column: {}'.format(column_shown))
        plt.xlabel('tensor_id_inside')
        plt.ylabel('Injection count where: '+column_shown+' is Nan')
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("No groups with NaN occurrences in column '{}'".format(column_shown))


def max_rel_err_percentage(injection_file):
    # Read the CSV file
    injection_data = pd.read_csv(injection_file)

    # Count the total number of injections for each algorithm pair
    total_injections = injection_data.groupby(['firstAlgorithmName', 'secondAlgorithmName']).size()

    # Filter rows where maxRelErr is greater than 0.01
    filtered_data = injection_data[injection_data['maxRelErr'] > 0.01]

    # Count the number of injections with maxRelErr > 0.01 for each algorithm pair
    error_injections = filtered_data.groupby(['firstAlgorithmName', 'secondAlgorithmName']).size()

    # Calculate the percentage of injections with maxRelErr > 0.01
    percentage_errors = (error_injections / total_injections * 100).fillna(0)

    if not percentage_errors.empty:
        # Plot the histogram
        percentage_errors.plot(kind='bar', figsize=(10, 6), title='Percentage of Max Relative Error > 0.01 between Algorithm Pairs')
        plt.xlabel('Algorithm Pairs')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("No occurrences of Max Relative Error > 0.01 found between algorithm pairs")

def simplify_algorithm_names(df):
    df['firstAlgorithmName'] = df['firstAlgorithmName'].apply(lambda x: 'GEMM' if 'GEMM' in x else 'FFT' if 'FFT' in x else 'WINOGRAD')
    df['secondAlgorithmName'] = df['secondAlgorithmName'].apply(lambda x: 'GEMM' if 'GEMM' in x else 'FFT' if 'FFT' in x else 'WINOGRAD')

def find_most_different_algorithm(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Simplify algorithm names
    simplify_algorithm_names(data)

    # Filter rows where maxRelErr > 0.01
    filtered_data = data[data['maxRelErr'] > 0.01]

    # Group by injectionId and find the record with minimum maxRelErr within each group
    min_error_data = filtered_data.loc[filtered_data.groupby('injectionId')['maxRelErr'].idxmin()]

    # Determine the most different algorithm
    algorithms = ['GEMM', 'FFT', 'WINOGRAD']
    most_different_counts = {alg: 0 for alg in algorithms}

    for _, row in min_error_data.iterrows():
        used_algorithms = {row['firstAlgorithmName'], row['secondAlgorithmName']}
        different_algorithm = (set(algorithms) - used_algorithms).pop()
        most_different_counts[different_algorithm] += 1

    # Plot the histogram
    plt.bar(most_different_counts.keys(), most_different_counts.values())
    plt.xlabel('Algorithm')
    plt.ylabel('Count of having the highest error - for each injection')
    plt.title('Algorithm with the highest deviation')
    plt.show()

def plot_injection_id_labels(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Simplify algorithm names
    simplify_algorithm_names(data)

    # Filter rows where maxRelErr > 0.01
    filtered_data = data[data['maxRelErr'] > 0.01]

    # Group by injectionId
    grouped = filtered_data.groupby('injectionId')

    # Initialize a dictionary to store the labels for each injectionId
    injection_labels = {}

    for injection_id, group in grouped:
        # Get unique algorithms in the current group
        unique_algorithms = set(group['firstAlgorithmName']).union(set(group['secondAlgorithmName']))
        # Check if all three algorithms are present
        if all(alg in unique_algorithms for alg in ['GEMM', 'FFT', 'WINOGRAD']):
            injection_labels[injection_id] = 1
        else:
            injection_labels[injection_id] = 0

    # Create a DataFrame from the labels dictionary
    labels_df = pd.DataFrame(list(injection_labels.items()), columns=['injectionId', 'Label'])

    # Plot the labels
    plt.scatter(labels_df['injectionId'], labels_df['Label'])
    plt.xlabel('Injection ID')
    plt.ylabel('Label')
    plt.title('Injection IDs with All Three Algorithms having maxRelErr > 0.01')
    plt.yticks([0, 1])
    plt.show()

def print_injection_id_labels(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Simplify algorithm names
    simplify_algorithm_names(data)

    # Filter rows where maxRelErr > 0.01
    filtered_data = data[data['maxRelErr'] > 0.01]

    # Group by injectionId
    grouped = filtered_data.groupby('injectionId')

    # Initialize a dictionary to store the labels for each injectionId
    injection_labels = {}

    for injection_id, group in grouped:
        # Get unique algorithms in the current group
        unique_algorithms = set(group['firstAlgorithmName']).union(set(group['secondAlgorithmName']))
        # Check if all three algorithms are present
        if all(alg in unique_algorithms for alg in ['GEMM', 'FFT', 'WINOGRAD']):
            injection_labels[injection_id] = 1
        else:
            injection_labels[injection_id] = 0

    # Create a DataFrame from the labels dictionary
    labels_df = pd.DataFrame(list(injection_labels.items()), columns=['injectionId', 'Label'])

    # Filter the injectionIds with label 1
    label_1_injection_ids = labels_df[labels_df['Label'] == 1]['injectionId'].tolist()

    # Print the count and list of injectionIds with label 1
    print(f"Number of injectionIds where all three algorithms diverge: {len(label_1_injection_ids)}")
    print("List of injectionIds where all three algorithms diverge:")
    print(label_1_injection_ids)

if __name__ == "__main__":
    fileFaultInj = 'FaultInjection.csv'
    fileFaultList = 'faultList.csv'

    nan_count(fileFaultInj, 'rootMediumSqErr')
    nan_on_bitpos(fileFaultInj, fileFaultList, 'rootMediumSqErr')
    nan_tensor_inside(fileFaultInj, fileFaultList, 'rootMediumSqErr')

    max_rel_err_percentage(fileFaultInj)
    find_most_different_algorithm(fileFaultInj)
    #plot_injection_id_labels(fileFaultInj)
    print_injection_id_labels(fileFaultInj)