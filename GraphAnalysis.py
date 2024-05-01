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

grouped_df.plot(kind='bar', y=['masked'], title='Count of Masked', ylabel='Count', xlabel='Layer', figsize=(10, 6))
grouped_df.plot(kind='bar', y=['non_critical'], title='Count of Non-Critical', ylabel='Count', xlabel='Layer', figsize=(10, 6))
grouped_df.plot(kind='bar', y=['critical'], title='Count of Critical', ylabel='Count', xlabel='Layer', figsize=(10, 6))

# Group by 'layer_injected' and aggregate the counts of 'critical'
grouped_df = df.groupby('layer_injected')['critical'].sum().reset_index()

# Sort the DataFrame by 'critical' column in descending order and select the top 10 layers
top_10_critical_layers = grouped_df.sort_values(by='critical', ascending=False).head(10)

# Plot the top 10 critical layers
top_10_critical_layers.plot(kind='bar', x='layer_injected', y='critical', title='Top 10 Critical Layers', ylabel='Count', xlabel='Layer', figsize=(10, 6))