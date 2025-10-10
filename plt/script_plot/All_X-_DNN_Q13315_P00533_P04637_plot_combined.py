import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14

# 读取三组数据并添加来源标签
df1 = pd.read_csv('/home/data1/BGM/fenziduiqi/All_X-_4179_DNN_Q13315_random_result/DNN_pearson_ratio_results.csv')
df2 = pd.read_csv('/home/data1/BGM/fenziduiqi/All_X-_4179_DNN_P00533_random_result/DNN_pearson_ratio_results.csv')
df3 = pd.read_csv('/home/data1/BGM/fenziduiqi/All_X-_4179_DNN_P04637_random_result/DNN_pearson_ratio_results.csv')

df1['Type'] = 'Q13315'
df2['Type'] = 'P00533'
df3['Type'] = 'P04637'

# 保留需要的列
df1 = df1[['Ratio', 'Pearson_Corr', 'Type']]
df2 = df2[['Ratio', 'Pearson_Corr', 'Type']]
df3 = df3[['Ratio', 'Pearson_Corr', 'Type']]

# 合并三组数据
df = pd.concat([df1, df2, df3], ignore_index=True)

# 自定义三种颜色（与Transformer风格一致）
box_colors = ['#75C8AE', '#E995C9', '#FC9871']  # Q13315, P00533, P04637

plt.figure(figsize=(24, 13))

sns.boxplot(
    x='Ratio', 
    y='Pearson_Corr', 
    hue='Type', 
    data=df, 
    palette=box_colors,
    boxprops=dict(alpha=0.8, linewidth=4),     
    fliersize=3, 
    width=0.9,   
    whiskerprops=dict(color='black', linewidth=3),  
    capprops=dict(color='black', linewidth=3),
    medianprops=dict(linewidth=4)  
)

plt.xlabel('Ratio', fontsize=58)
plt.ylabel('Pearson', fontsize=58)
plt.title('Pearson by Ratio of DNN', fontsize=58)
plt.tick_params(axis='both', labelsize=56)
plt.ylim(-0.3, 0.6)
plt.yticks([-0.2, 0, 0.2, 0.4, 0.6], fontsize=56)
plt.legend(title='', fontsize=44, loc='upper left')

for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(4)

plt.tight_layout()
plt.savefig('/home/data1/BGM/fenziduiqi/All_X-_4179_DNN_Q13315_P00533_P04637_plot_pearson_combined.png', dpi=900)
plt.close()
