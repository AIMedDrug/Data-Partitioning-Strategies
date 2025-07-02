import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14

# Load CSV file
csv_file = '/home/data1/BGM/fenziduiqi/All_X-_SVR_4179_result/SVR_results.csv'
df = pd.read_csv(csv_file)

# Ensure only needed columns are used
df = df[['C', 'Epsilon', 'Kernel', 'Pearson Corr']]

# Define custom color palette for Epsilon and Kernel
epsilon_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])
kernel_colors = sns.color_palette(['skyblue', 'palegreen', 'lightpink', 'lightyellow', 'red'])

# Create two subplots
plt.figure(figsize=(15, 20))

# First subplot: Group by C, Epsilon as hue
plt.subplot(2, 1, 1)
sns.boxplot(x='C', y='Pearson Corr', hue='Epsilon', data=df, palette=epsilon_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

plt.legend(fontsize=36, title='Epsilon', title_fontsize=40)
plt.xlabel('C', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by C and Epsilon of SVR', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.1, 0.8)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=48)

# Set spines to black and bold
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

# Second subplot: Group by C, Kernel as hue
plt.subplot(2, 1, 2)
sns.boxplot(x='C', y='Pearson Corr', hue='Kernel', data=df, palette=kernel_colors, 
            boxprops=dict(alpha=0.8), fliersize=3, width=0.8,
            whiskerprops=dict(color='black', linewidth=2),  # Set whiskers to black and bold
            capprops=dict(color='black', linewidth=2))      # Set caps to black and bold

plt.legend(fontsize=36, title='Kernel', title_fontsize=40)
plt.xlabel('C', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by C and Kernel of SVR', fontsize=50)
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
plt.savefig('/home/data1/BGM/fenziduiqi/All_X-_4179_SVR_plot_pearson_random.png', dpi=900)
plt.close()