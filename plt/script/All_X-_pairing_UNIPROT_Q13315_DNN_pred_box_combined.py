import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14

# 读取两组数据
df1 = pd.read_csv('/home/data1/BGM/BGM_fenziduiqi/All_X-_4179_DNN_Q13315_random_pearson_ratio0.1~0.9_results.csv')
df2 = pd.read_csv('/home/data1/BGM/BGM_fenziduiqi/All_X-_Data_pairing_UNIPROT_Q13315_DNN_pred_Calculate_Pearson.csv')

# 添加来源标签
df1['Type'] = 'Unpaired'
df2['Type'] = 'Paired'

# 保留需要的列
df1 = df1[['Ratio', 'Pearson_Corr', 'Type']]
df2 = df2[['Ratio', 'Pearson_Corr', 'Type']]

# 合并
df = pd.concat([df1, df2], ignore_index=True)

# 自定义颜色
box_colors = ['#4C8BE6', '#E995C9']  # Unpaired, Paired

plt.figure(figsize=(18, 11))

sns.boxplot(
    x='Ratio', y='Pearson_Corr', hue='Type', data=df,
    palette=box_colors,
    boxprops=dict(alpha=0.8),
    fliersize=3, width=0.7,
    whiskerprops=dict(color='black', linewidth=2),
    capprops=dict(color='black', linewidth=2)
)

plt.xlabel('Ratio', fontsize=50)
plt.ylabel('Pearson', fontsize=50)
plt.title('Pearson by Ratio of DNN', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.2, 0.5)
plt.yticks([-0.1, 0.1, 0.3, 0.5], fontsize=48)
plt.legend(title='', fontsize=44, loc='upper left')

for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('All_X-_Data_pairing_UNIPROT_Q13315_DNN_pred_box_pearson_combined.png', dpi=900)
plt.close()
