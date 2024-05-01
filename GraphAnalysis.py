import pandas as pd

df = pd.read_csv('injection_output.csv')

# Group by 'layer_injected' and aggregate the counts of 'masked', 'non_critical', and 'critical'
grouped_df = df.groupby('layer_injected').agg({
    'masked': 'sum',
    'non_critical': 'sum',
    'critical': 'sum'
})

# Print the grouped DataFrame
print(grouped_df)

