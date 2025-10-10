import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score
import glob
import os

# ========= 路径 =========
base_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pairing_UNIPROT_P00520_p00519_RF_pred_result'

# 查找所有子文件夹中的 query_pred_ddG_means.csv 文件
file_paths = glob.glob(os.path.join(base_dir, '*/query_pred_ddG_means.csv'))
if len(file_paths) != 30:
    print(f"Warning: Found {len(file_paths)} files, expected 30")

# 读取所有CSV的 q_real 列，取均值作为实验值
dataframes = [pd.read_csv(file) for file in file_paths]
mean_boot_y_true = np.mean([df['q_real'].values for df in dataframes], axis=0)

# ========= 人为设定预测值全为0 =========
mean_boot_y_pred = np.zeros_like(mean_boot_y_true)

# ========= 阈值 =========
threshold_accuracy = 1.36
mean_pred_labels = (mean_boot_y_pred > threshold_accuracy).astype(int)
mean_real_labels = (mean_boot_y_true > threshold_accuracy).astype(int)

# ========= 混淆矩阵 =========
cm = confusion_matrix(mean_real_labels, mean_pred_labels)

plt.figure(figsize=(6.8, 6.8))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greys',
    cbar=False,
    xticklabels=['S', 'R'],
    yticklabels=['S', 'R'],
    annot_kws={"size": 44, "color": "black"},
    square=True
)

# 自定义格子颜色
colors = [
    ['#2166ACFF', '#92C5DEFF'],
    ['#F4A582FF', '#FDDBC7FF']
]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.add_patch(
            plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i][j], alpha=0.8, lw=0)
        )

# 标签和刻度
ax.set_xlabel('Prediction', fontsize=40)
ax.set_ylabel('Experiment', fontsize=40)
ax.set_xticklabels(['S', 'R'], fontsize=42)
ax.set_yticklabels(['S', 'R'], fontsize=42, rotation=0)

ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_top()

plt.tight_layout()
plt.savefig("plot_matrix_pred0_AUC.png", dpi=900)
plt.show()

# ========= 计算 AUC =========
try:
    auc_value = roc_auc_score(mean_real_labels, mean_boot_y_pred)
except ValueError:
    auc_value = float("nan")

print(f"AUC: {auc_value}")
