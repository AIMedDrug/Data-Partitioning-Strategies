import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14

# Load CSV file
csv_file = '/home/data1/BGM/fenziduiqi/All_X-_UNIPORT_4179_GRU_result/GRU_UNIPORT_results.csv'
df = pd.read_csv(csv_file)

# Ensure only needed columns are used
df = df[['Hidden Dim', 'Dropout', 'Num Layers', 'Pearson Corr']]

# Define custom color palette for Dropout and Num Layers
dropout_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])
num_layers_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])

# Create two subplots
plt.figure(figsize=(15, 20))

# First subplot: Group by Hidden Dim, Dropout as hue
plt.subplot(2, 1, 1)
sns.boxplot(x='Hidden Dim', y='Pearson Corr', hue='Dropout', data=df, palette=dropout_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

# Create custom legend handles
handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().get_legend().remove()  # Remove default legend

# Add custom legend inside the plot using provided coordinates
x_positions = [0.93, 0.81, 0.93, 0.81, 0.93]
y_positions = [0.92, 0.86, 0.86, 0.80, 0.80]
for i, (handle, label) in enumerate(zip(handles, labels)):
    plt.figtext(x_positions[i], y_positions[i], label, 
                ha='center', va='center', fontsize=36,
                bbox=dict(boxstyle="square,pad=0.44", facecolor=handle.get_facecolor(), alpha=0.8, edgecolor='none'))

plt.figtext(0.8, 0.92, 'Dropout', ha='center', va='center', fontsize=40)

plt.xlabel('Hidden Dim', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by HD and Dt of GRU', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.1, 0.4)
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Second subplot: Group by Hidden Dim, Num Layers as hue
plt.subplot(2, 1, 2)
sns.boxplot(x='Hidden Dim', y='Pearson Corr', hue='Num Layers', data=df, palette=num_layers_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

plt.legend(fontsize=36, title='Num Layers', title_fontsize=40)
plt.xlabel('Hidden Dim', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by HD and NL of GRU', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.1, 0.4)
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/data1/BGM/fenziduiqi/All_X-_4179_GRU_plot_pearson_UNIPORT.png', dpi=900)
plt.close()