import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 14  # 默认字体

# 1. 读入数据
excel_path = "/home/data1/BGM/fenziduiqi/拼接和差值all_model_best_uniport_pearson.xlsx"
save_path = "/home/data1/BGM/fenziduiqi/All_esm_X-_model_plot_uniport_box.png"
models = ["RF", "SVR", "GRU", "BiLSTM", "DNN", "TF"]

df = pd.read_excel(excel_path, usecols=range(3, 15))
df = df.apply(pd.to_numeric, errors="coerce")

# 2. 整理数据，确保按[Con, Dif, Con, Dif, ...]交替读取
data = []
for i, m in enumerate(models):
    con_vals = df.iloc[:, 2*i].dropna().values    # Con列
    dif_vals = df.iloc[:, 2*i+1].dropna().values  # Dif列
    data.extend([{'Model': m, 'Group': 'Con', 'Pearson': v} for v in con_vals])
    data.extend([{'Model': m, 'Group': 'Dif', 'Pearson': v} for v in dif_vals])
plot_df = pd.DataFrame(data)

# 3. 配色
custom_colors = ['#B0DC66', '#E995C9']
palette = {'Con': custom_colors[0], 'Dif': custom_colors[1]}

# 4. 绘图
plt.figure(figsize=(15, 12))
plt.subplots_adjust(top=0.93, bottom=0.17, left=0.17, right=0.98)
ax = sns.boxplot(
    x='Model', y='Pearson', hue='Group', data=plot_df,
    palette=palette,
    boxprops=dict(alpha=0.8),
    fliersize=3,
    width=0.6,
    whiskerprops=dict(color='black', linewidth=2),
    capprops=dict(color='black', linewidth=2),
    medianprops=dict(color='black', linewidth=2),
    linewidth=2,
)

plt.xlabel("")  # 确保x轴没有标题
plt.ylabel('Pearson', fontsize=50)
plt.title('Con VS Dif - UNIPROT', fontsize=50)
plt.tick_params(axis='both', labelsize=48)
plt.ylim(-0.35, 0.5)
plt.yticks([-0.3, -0.1, 0.1, 0.3, 0.5], fontsize=48)

# 设置spines为黑色和加粗
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xticks(rotation=30)
plt.legend(title='', fontsize=42, loc='upper right', title_fontsize=40)
plt.savefig(save_path, dpi=900)
plt.close()
