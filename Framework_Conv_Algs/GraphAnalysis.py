import pandas as pd
import matplotlib.pyplot as plt

def nan_count(file_path, column_name):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Create histogram
    nan_counts = data.groupby([data['firstAlgorithmName'], data['secondAlgorithmName']])[column_name].apply(lambda x: x.isna().sum())
    nan_counts.plot(kind='bar', figsize=(10, 6), title='NaN Count in Column: {}'.format(column_name))
    plt.xlabel('Algorithm Names')
    plt.ylabel('NaN Count')
    plt.xticks(rotation=90)
    plt.show()

def average_err(file_path, column_name):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Drop rows with NaN values in the specified column
    data_cleaned = data.dropna(subset=[column_name])

    # Calculate the average for each group
    grouped_avg = data_cleaned.groupby([data_cleaned['firstAlgorithmName'], data_cleaned['secondAlgorithmName']])[column_name].mean()
    grouped_avg.plot(kind='bar', figsize=(10, 6), title='Average of Column: {}'.format(column_name))
    plt.xlabel('Algorithm Names')
    plt.ylabel('Average')
    plt.xticks(rotation=90)
    plt.show()

def nan_on_injid(injection_file, fault_file, column_name):
    nan_count_utils(injection_file, fault_file, 'injectionId', column_name, True)

def nan_on_bitpos(injection_file, fault_file, column_name):
    nan_count_utils(injection_file, fault_file, 'bit_pos', column_name, False)

def nan_on_tensor(injection_file, fault_file, column_name):
    nan_count_utils(injection_file, fault_file, 'tensor_id', column_name, True)

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
        plt.ylabel('NaN Count')
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
        plt.ylabel('NaN Count')
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("No groups with NaN occurrences in column '{}'".format(column_shown))



if __name__ == "__main__":
    fileFaultInj = 'FaultInjection.csv'
    fileFaultList = 'faultList.csv'

    nan_count(fileFaultInj, 'rootMediumSqErr')
    average_err(fileFaultInj, 'rootMediumSqErr')
    average_err(fileFaultInj, 'maxRelErr')
    nan_on_injid(fileFaultInj, fileFaultList, 'rootMediumSqErr')
    nan_on_bitpos(fileFaultInj, fileFaultList, 'rootMediumSqErr')
    nan_on_tensor(fileFaultInj, fileFaultList, 'rootMediumSqErr')
    nan_tensor_inside(fileFaultInj, fileFaultList, 'rootMediumSqErr')