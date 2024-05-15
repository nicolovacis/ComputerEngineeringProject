import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

def dim_tensor_correlation(file, WEIGHTS_PATH):
    weights = torch.load(WEIGHTS_PATH)['model'].state_dict()

    df = pd.read_csv(file)

    # Initialize a dictionary to store critical injections for each dimension
    dimension_critical = {}

    for layer_name in df['layer_injected'].unique():

        # Tensor dimension
        tensorWeights = weights.get(layer_name)
        tensorWeights_size_list = list(tensorWeights.size()) # list format of the tensor size --> [4,2,3,5]
        tensor_dim = 1
        for num in tensorWeights_size_list:
            tensor_dim *= num

        # Calculate total injections for the current layer
        total_injections = df[df['layer_injected'] == layer_name]['masked'].sum() + \
                           df[df['layer_injected'] == layer_name]['non_critical'].sum() + \
                           df[df['layer_injected'] == layer_name]['critical'].sum()

        # Calculate critical injections for the current layer
        critical_injections = df[df['layer_injected'] == layer_name]['critical'].sum()
        percentage_critical = (critical_injections / total_injections) * 100

        # Store the percentage of critical injections for the current dimension
        if tensor_dim not in dimension_critical:
            dimension_critical[tensor_dim] = []
        dimension_critical[tensor_dim].append(percentage_critical)

    # Calculate the mean percentage of critical injections for each dimension
    dimension_mean_critical = {dim: np.mean(perc) for dim, perc in dimension_critical.items()}

    # Plot a graph (Commented out)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(dimension_mean_critical.keys(), dimension_mean_critical.values())
    # plt.xlabel('Dimension of Tensor')
    # plt.ylabel('Percentage of Critical Injections')
    # plt.title('Correlation between Tensor Dimension and Critical Injections')
    # plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    # plt.grid(True)
    # plt.show()

def injection_types_plot(file):

    df = pd.read_csv(file)

    grouped_df = df.groupby('layer_injected').agg({
        'masked': 'sum',
        'non_critical': 'sum',
        'critical': 'sum'
    })

    total_injections = grouped_df['masked'] + grouped_df['non_critical'] + grouped_df['critical']

    grouped_df['critical_percent'] = (grouped_df['critical'] / total_injections) * 100

    grouped_df.drop(columns=['masked', 'non_critical', 'critical'], inplace=True)

    grouped_df.plot(kind='bar', y='critical_percent', title='Percentage of Critical Injections', ylabel='Percentage', xlabel='Layer', figsize=(10, 6))

def vulnerability_plot(type, file):
    df = pd.read_csv(file)

    if type == "layer":

        grouped_df = df.groupby('layer_injected').agg({
            'masked': 'sum',
            'non_critical': 'sum',
            'critical': 'sum'
        })

        total_injections = grouped_df.sum(axis=1)

        grouped_df['critical_percentage'] = (grouped_df['critical'] / total_injections) * 100

        top_critical_layers = grouped_df.sort_values(by='critical_percentage', ascending=False).head(10)

        top_critical_layers.plot(kind='bar', y='critical_percentage', title='Vulnerable Layers', ylabel='Critical Injection Percentage (out of all the injections)', xlabel='Layer', figsize=(10, 6))

    elif type == 'bit':

        df['bit_to_change'] = pd.to_numeric(df['bit_to_change'])

        grouped_df = df.groupby('bit_to_change').agg({
            'masked': 'sum',
            'non_critical': 'sum',
            'critical': 'sum'
        })

        total_injections_per_bit = grouped_df.sum(axis=1)

        grouped_df['critical_percentage'] = (grouped_df['critical'] / total_injections_per_bit) * 100
        grouped_df['masked_percentage'] = (grouped_df['masked'] / total_injections_per_bit) * 100
        grouped_df['non_critical_percentage'] = (grouped_df['non_critical'] / total_injections_per_bit) * 100

        grouped_df['total_percentage'] = grouped_df['critical_percentage'] + grouped_df['masked_percentage'] + grouped_df['non_critical_percentage']
        grouped_df['non_critical_percentage'] += 100 - grouped_df['total_percentage']

        grouped_df.drop(columns=['total_percentage'], inplace=True)

        # to maintain the numerical order
        grouped_df = grouped_df.sort_index()

        grouped_df.drop(columns=['masked', 'non_critical', 'critical'], inplace=True)

        grouped_df.plot(kind='bar', stacked=True, title='Vulnerable Bits', ylabel='Critical Injection Percentage (out of all the injections)', xlabel='Bit', figsize=(10, 6))


if __name__ == "__main__":

    #COLAB PATH (testing):
    #file_path = 'injection_output.csv'
    #weight_path = '/content/drive/MyDrive/Colab Notebooks/vit_iiipet_train_best.pth'

    parser = argparse.ArgumentParser(
        prog='GraphAnalysis',
        description='Script that generates graph to analyse the injections\' output ',
    )

    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)

    args = parser.parse_args()

    file_path = args.file_path
    weight_path = args.weight_path

    #dim_tensor_correlation(file_path, weight_path)
    injection_types_plot(file_path)
    vulnerability_plot("layer", file_path)
    vulnerability_plot("bit", file_path)