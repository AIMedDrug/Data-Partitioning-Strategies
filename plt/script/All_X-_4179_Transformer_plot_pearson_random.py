import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14

# Load CSV file
csv_file = '/home/data1/BGM/fenziduiqi/All_X-_Transformer_4179_result/Transformer_results.csv'
df = pd.read_csv(csv_file)

# Ensure only needed columns are used
df = df[['Hidden Dim', 'Dropout', 'Num Layers', 'Nhead', 'Pearson Corr']]

# Define custom color palette for Dropout and Nhead
dropout_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])
nhead_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])

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
x_positions = [0.93, 0.8, 0.93, 0.8, 0.93]
y_positions = [0.705, 0.655, 0.655, 0.605, 0.605]
for i, (handle, label) in enumerate(zip(handles, labels)):
    plt.figtext(x_positions[i], y_positions[i], label, 
                ha='center', va='center', fontsize=36,
                bbox=dict(boxstyle="square,pad=0.44", facecolor=handle.get_facecolor(), alpha=0.8, edgecolor='none'))

plt.figtext(0.8, 0.705, 'Dropout', ha='center', va='center', fontsize=40)

plt.xlabel('Hidden Dim', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by HD and Dt of TF', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.1, 0.8)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Second subplot: Group by Hidden Dim, Nhead as hue
plt.subplot(2, 1, 2)
sns.boxplot(x='Hidden Dim', y='Pearson Corr', hue='Nhead', data=df, palette=nhead_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

plt.legend(fontsize=36, title='Nhead', title_fontsize=40)
plt.xlabel('Hidden Dim', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by HD and Nhead of TF', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.1, 0.8)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('/home/data1/BGM/fenziduiqi/All_X-_4179_Transformer_plot_pearson_random.png', dpi=900)
plt.close()