import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


def adjust_x_labels(label):
    if label.startswith("encoder.layers."):
        label = label.replace("encoder.layers.", "")
    if label.endswith(".weight"):
        label = label.replace(".weight", "")
    return label

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

    grouped_df.plot(kind='bar', y='critical_percent', title='Percentage of Critical Injections', ylabel='Critical Injection Percentage (out of all the injections)', xlabel='Layer injected', figsize=(10, 6))

    plt.gca().set_xticklabels(map(adjust_x_labels, grouped_df.index), rotation=80, ha='right')
    plt.show()


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

        plt.gca().set_xticklabels(map(adjust_x_labels, top_critical_layers.index), rotation=45, ha='right')
        plt.show()

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

        grouped_df.plot(kind='bar', stacked=True, title='Vulnerable Bits', ylabel='Injection types - percentage', xlabel='Bit flipped', figsize=(10, 6))


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