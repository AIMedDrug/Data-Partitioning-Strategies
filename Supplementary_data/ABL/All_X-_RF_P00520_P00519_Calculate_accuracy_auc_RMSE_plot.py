import pandas as pd
import numpy as np
from cuml.metrics import accuracy_score, roc_auc_score  # 使用cuml进行GPU加速
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
import seaborn as sns
from scipy.stats import linregress, pearsonr
from scipy.stats import weightedtau, kendalltau

# 包含query_pred_ddG_means.csv文件的子文件夹的目录
base_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pairing_UNIPROT_P00520_p00519_RF_pred_result'

# 查找所有子文件夹中的query_pred_ddG_means.csv文件
file_paths = glob.glob(os.path.join(base_dir, '*/query_pred_ddG_means.csv'))
if len(file_paths) != 30:
    print(f"Warning: Found {len(file_paths)} files, expected 30")

# 将所有CSV文件加载为DataFrame列表
dataframes = [pd.read_csv(file) for file in file_paths]
# 检查每个文件是否有127行
for i, df in enumerate(dataframes):
    if len(df) != 127:
        print(f"Warning: File {file_paths[i]} has {len(df)} rows, expected 127")

# Bootstrap参数
n_bootstraps = 1000
n_rows = 127  # 每个文件的行数
n_files = len(dataframes)
boot_accuracy = []
boot_roc_auc = []
boot_rmse = []
threshold_accuracy = 1.36
threshold_roc_auc = 0.56  # 假设这是用于roc_auc_score的阈值

# Bootstrap重采样
for _ in range(n_bootstraps):
    # 通过为每个行索引(0-126)随机选择一个文件，创建一个新数据集
    boot_indices = np.random.randint(0, n_files, size=n_rows)  # 为每一行随机选择文件
    boot_y_true = np.zeros(n_rows)
    boot_y_pred = np.zeros(n_rows)
    
    for row_idx in range(n_rows):
        file_idx = boot_indices[row_idx]
        boot_y_true[row_idx] = dataframes[file_idx]['q_real'].iloc[row_idx]
        boot_y_pred[row_idx] = dataframes[file_idx]['q_pred_mean'].iloc[row_idx]
    
    # 计算准确率(阈值=1.36)
    pred_labels = (boot_y_pred > threshold_accuracy).astype(int)
    real_labels = (boot_y_true > threshold_accuracy).astype(int)
    boot_accuracy.append(accuracy_score(real_labels, pred_labels))
    
    # 计算ROC AUC(使用q_pred_mean作为概率)
    boot_roc_auc.append(roc_auc_score(real_labels, boot_y_pred))
    
    # 计算RMSE(使用原始值)
    boot_rmse.append(np.sqrt(mean_squared_error(boot_y_true, boot_y_pred)))

# 计算均值和95%置信区间
accuracy_mean = np.mean(boot_accuracy)
accuracy_ci = np.percentile(boot_accuracy, [2.5, 97.5])
roc_auc_mean = np.mean(boot_roc_auc)
roc_auc_ci = np.percentile(boot_roc_auc, [2.5, 97.5])
rmse_mean = np.mean(boot_rmse)
rmse_ci = np.percentile(boot_rmse, [2.5, 97.5])

# 打印结果
print(f"Accuracy (threshold={threshold_accuracy}): {accuracy_mean:.4f}, 95% CI: ({accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f})")
print(f"ROC AUC: {roc_auc_mean:.4f}, 95% CI: ({roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f})")
print(f"RMSE: {rmse_mean:.4f}, 95% CI: ({rmse_ci[0]:.4f}, {rmse_ci[1]:.4f})")

# 绘图：带回归线的散点图
# 使用bootstrap均值预测进行绘图
mean_boot_y_true = np.mean([dataframes[i]['q_real'].values for i in range(n_files)], axis=0)
mean_boot_y_pred = np.mean([dataframes[i]['q_pred_mean'].values for i in range(n_files)], axis=0)

# 读取FreeEnergyResults.dat文件，提取DDG_exp和DDG_fep1,2,3的均值
free_energy_file = '/home/bioinfor6/BGM/fenziduiqi/FreeEnergyResults.dat'
# 用delim_whitespace自动识别空格或tab分隔，防止解析报错
free_energy_df = pd.read_csv(free_energy_file, sep=r'\s+')
# 过滤掉缺失行
free_energy_df = free_energy_df.dropna(subset=['DDG_exp', 'DDG_fep1', 'DDG_fep2', 'DDG_fep3'])
free_energy_x = free_energy_df['DDG_exp'].astype(float)
free_energy_y = free_energy_df[['DDG_fep1', 'DDG_fep2', 'DDG_fep3']].astype(float).mean(axis=1)

r, p = kendalltau(free_energy_x, free_energy_y)
print("kendaltau of FEP:", r)
r, p = kendalltau(mean_boot_y_true, mean_boot_y_pred)
print("kendaltau of RF:", r)

r, p = pearsonr(free_energy_x, free_energy_y)
print("pearsonr of FEP:", r)
r, p = pearsonr(mean_boot_y_true, mean_boot_y_pred)
print("pearsonr of RF:", r)

plt.figure(figsize=(14, 12))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)
plt.subplots_adjust(top=0.93, bottom=0.13, left=0.17, right=0.96)

# 先画FEP散点（底层）
plt.scatter(
    free_energy_x, free_energy_y,
    facecolor='#FFEBA6', edgecolor='#239B3A', s=400, linewidths=3, label='FEP', zorder=1
)

# 画回归线和置信区间（置信区间在RF点下层）
slope, intercept, r_value, p_value, std_err = linregress(mean_boot_y_true, mean_boot_y_pred)
line = slope * mean_boot_y_true + intercept
# 画置信区间
sns.regplot(x=free_energy_x, y=free_energy_y, ax=ax, scatter=False, color='#FFEBA6')
#plt.fill_between(mean_boot_y_true, line - std_err, line + std_err, color='yellow', zorder=4)
# 画回归线
#plt.plot(mean_boot_y_true, line, color='#E52D07', linestyle='-', alpha=0.7, linewidth=5, zorder=3)
# 再画RF散点（上层）
plt.scatter(
    mean_boot_y_true, mean_boot_y_pred,
    facecolor='#2D8BC1', edgecolor='#754D98', s=400, linewidths=3, label='RF', zorder=2
)

sns.regplot(x=mean_boot_y_true, y=mean_boot_y_pred, ax=ax, scatter=False, color='#2D8BC1')
plt.xlabel(r'$\Delta G_{Exp.}$ (kcal mol$^{-1}$)', fontsize=49)
plt.ylabel(r'$\Delta G_{pred}$ (kcal mol$^{-1}$)', fontsize=49)
plt.title('ABL Kinase', fontsize=50)
plt.xlim(-3, 6)
plt.xticks([ -2, 0, 2, 4, 6], fontsize=46)
plt.ylim(-3, 6)
plt.yticks([ -2, 0, 2, 4, 6], fontsize=46)
plt.grid(False)
plt.legend(fontsize=36, loc='upper left')
output_path_scatter = '/home/bioinfor6/BGM/fenziduiqi/All_X-_4179_RF_P00520_P00519_Calculate_accuracy_roc_auc_score_RMSE_result/scatter_plot.png'
plt.savefig(output_path_scatter, dpi=900)
plt.show()

# 绘图：混淆矩阵
# 使用均值阈值标签绘制混淆矩阵
mean_pred_labels = (mean_boot_y_pred > threshold_accuracy).astype(int)
mean_real_labels = (mean_boot_y_true > threshold_accuracy).astype(int)
cm = confusion_matrix(mean_real_labels, mean_pred_labels)
plt.figure(figsize=(6.8, 6.8))  # 这里设置整个图像为正方形
ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greys',
    cbar=False,
    xticklabels=['S', 'R'],
    yticklabels=['S', 'R'],
    annot_kws={"size": 44, "color": "black"},  # 设置数字大小和颜色为黑色
    square=True  # 设置格子为正方形
)

# 自定义每个格子的颜色
colors = [
    ['#2166ACFF', '#92C5DEFF'],
    ['#F4A582FF', '#FDDBC7FF']
]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.add_patch(
            plt.Rectangle(
                (j, i), 1, 1, fill=True, color=colors[i][j], alpha=0.8, lw=0
            )
        )

ax.set_xlabel('Prediction', fontsize=40)
ax.set_ylabel('Experiment', fontsize=40)
ax.set_xticklabels(['S', 'R'], fontsize=42)
ax.set_yticklabels(['S', 'R'], fontsize=42, rotation=0)

ax.xaxis.set_label_position('bottom')  # x轴标签在下方
ax.xaxis.tick_top()                 # x轴刻度在上方

plt.tight_layout()
output_path_cm = '/home/bioinfor6/BGM/fenziduiqi/All_X-_4179_RF_P00520_P00519_Calculate_accuracy_roc_auc_score_RMSE_result/confusion_matrix.png'
plt.savefig(output_path_cm, dpi=900)
plt.show()
