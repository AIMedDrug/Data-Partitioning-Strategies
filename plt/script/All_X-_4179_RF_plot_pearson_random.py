import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14

# Load CSV file
csv_file = '/home/data1/BGM/fenziduiqi/All_X-_RF_4179_result/RF_results.csv'
df = pd.read_csv(csv_file)

# Ensure only needed columns are used
df = df[['N Estimators', 'Max Depth', 'Min Samples Split', 'Pearson Corr']]

# Define custom color palette for Max Depth and Min Samples Split
max_depth_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])
min_samples_split_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])

# Create two subplots
plt.figure(figsize=(15, 20))

# First subplot: Group by N Estimators, Max Depth as hue
plt.subplot(2, 1, 1)
sns.boxplot(x='N Estimators', y='Pearson Corr', hue='Max Depth', data=df, palette=max_depth_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

plt.legend(fontsize=36, title='Max Depth', title_fontsize=40)
plt.xlabel('N Estimators', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by NE and MD of RF', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.2, 0.8)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Second subplot: Group by N Estimators, Min Samples Split as hue
plt.subplot(2, 1, 2)
sns.boxplot(x='N Estimators', y='Pearson Corr', hue='Min Samples Split', data=df, palette=min_samples_split_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

plt.legend(fontsize=36, title='MS Split', title_fontsize=40)
plt.xlabel('N Estimators', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by NE and MS Split of RF', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.2, 0.8)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/data1/BGM/fenziduiqi/All_X-_4179_RF_plot_pearson_random.png', dpi=900)
plt.close()