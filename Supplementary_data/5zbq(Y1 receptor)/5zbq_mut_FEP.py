import os
import glob
import pandas as pd
import numpy as np
from cuml.metrics import accuracy_score, roc_auc_score  # GPU加速
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress, pearsonr, kendalltau, spearmanr  # 新增 spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ================== RF 部分 ==================
base_dir = '/home/bioinfor6/BGM/fenziduiqi/5zbq/RF_pred_result_unpaired/n100_d20_s10'

# 查找所有子文件夹中的 RF_prediction_results.csv 文件
file_paths = glob.glob(os.path.join(base_dir, 'result_*/RF_prediction_results.csv'))
print(f"Found {len(file_paths)} files.")

# 将所有 CSV 文件加载为 DataFrame 列表
dataframes = [pd.read_csv(file) for file in file_paths]

# 平均 RF 的预测值和真实值
mean_boot_y_true = dataframes[0]['True_ddG'].values  # 真实差异
mean_boot_y_pred = np.mean([df['Predicted_ddG'].values for df in dataframes], axis=0)  # 预测均值

# ================== FEP 部分（只取 label=test） ==================
fep_file = '/home/bioinfor6/BGM/fenziduiqi/5zbq_mutation_fep_with_sequences.csv'
fep_df = pd.read_csv(fep_file, usecols=["mutation", "FEP", "Exp", "label"])

# 只选择 label=='test' 的数据
fep_test_df = fep_df[fep_df['label'] == 'test']

fep_x = fep_test_df["Exp"].astype(float)
fep_y = fep_test_df["FEP"].astype(float)

# ================== 相关性计算 ==================
# Kendall tau
r_kendall_fep, _ = kendalltau(fep_x, fep_y)
r_kendall_rf, _ = kendalltau(mean_boot_y_true, mean_boot_y_pred)
print("kendaltau of FEP (label=test):", r_kendall_fep)
print("kendaltau of RF:", r_kendall_rf)

# Pearson r
r_pearson_fep, _ = pearsonr(fep_x, fep_y)
r_pearson_rf, _ = pearsonr(mean_boot_y_true, mean_boot_y_pred)
print("pearsonr of FEP (label=test):", r_pearson_fep)
print("pearsonr of RF:", r_pearson_rf)

# Spearman rho
r_spearman_fep, _ = spearmanr(fep_x, fep_y)
r_spearman_rf, _ = spearmanr(mean_boot_y_true, mean_boot_y_pred)
print("spearmanr of FEP (label=test):", r_spearman_fep)
print("spearmanr of RF:", r_spearman_rf)

# ================== RMSE（可选加载汇总文件） ==================
summary_file = os.path.join(base_dir, 'test_samples_pearson_rmse_spearman_summary.csv')
if os.path.exists(summary_file):
    summary_df = pd.read_csv(summary_file)
    mean_rmse_rf = summary_df['RMSE'].mean()
    print(f"Mean RMSE of RF: {mean_rmse_rf:.4f}")

# ================== Bootstrap 采样（应用于 RF 部分） ==================
n_bootstraps = 1000
n_rows = len(dataframes[0]) if dataframes else 0
n_files = len(dataframes)
boot_accuracy = []
boot_roc_auc = []
boot_rmse = []
threshold_accuracy = 1.36  # 二分类阈值

if n_rows > 0 and n_files > 0:
    for _ in range(n_bootstraps):
        boot_indices = np.random.randint(0, n_files, size=n_rows)
        boot_y_true = np.zeros(n_rows)
        boot_y_pred = np.zeros(n_rows)
        
        for row_idx in range(n_rows):
            file_idx = boot_indices[row_idx]
            boot_y_true[row_idx] = dataframes[file_idx]['True_ddG'].iloc[row_idx]
            boot_y_pred[row_idx] = dataframes[file_idx]['Predicted_ddG'].iloc[row_idx]
        
        # 准确率
        pred_labels = (boot_y_pred > threshold_accuracy).astype(int)
        real_labels = (boot_y_true > threshold_accuracy).astype(int)
        boot_accuracy.append(accuracy_score(real_labels, pred_labels))
        
        # ROC AUC
        boot_roc_auc.append(roc_auc_score(real_labels, boot_y_pred))
        
        # RMSE
        boot_rmse.append(np.sqrt(mean_squared_error(boot_y_true, boot_y_pred)))

    # 计算均值与95%置信区间
    accuracy_mean = np.mean(boot_accuracy)
    accuracy_ci = np.percentile(boot_accuracy, [2.5, 97.5])
    roc_auc_mean = np.mean(boot_roc_auc)
    roc_auc_ci = np.percentile(boot_roc_auc, [2.5, 97.5])
    rmse_mean = np.mean(boot_rmse)
    rmse_ci = np.percentile(boot_rmse, [2.5, 97.5])

    print(f"Accuracy (threshold={threshold_accuracy}): {accuracy_mean:.4f}, 95% CI: ({accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f})")
    print(f"ROC AUC: {roc_auc_mean:.4f}, 95% CI: ({roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f})")
    print(f"RMSE: {rmse_mean:.4f}, 95% CI: ({rmse_ci[0]:.4f}, {rmse_ci[1]:.4f})")
else:
    print("No data for Bootstrap sampling.")

# ================== 绘制散点图 ==================
plt.figure(figsize=(14, 12))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)
plt.subplots_adjust(top=0.93, bottom=0.13, left=0.17, right=0.96)

# FEP 散点
plt.scatter(
    fep_x, fep_y,
    facecolor='#c2c2c2', edgecolor='#845116', s=400, linewidths=3, label='FEP', zorder=1
)
sns.regplot(x=fep_x, y=fep_y, ax=ax, scatter=False, color='#FFEBA6')

# RF 散点
plt.scatter(
    mean_boot_y_true, mean_boot_y_pred,
    facecolor='#2D8BC1', edgecolor='#754D98', s=400, linewidths=3, label='RF', zorder=2
)
sns.regplot(x=mean_boot_y_true, y=mean_boot_y_pred, ax=ax, scatter=False, color='#2D8BC1')

# y=0 红色虚线
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, zorder=6)

plt.xlabel(r'$\Delta G_{Exp.}$ (kcal mol$^{-1}$)', fontsize=49)
plt.ylabel(r'$\Delta G_{pred}$ (kcal mol$^{-1}$)', fontsize=49)
plt.title('Y1 receptor', fontsize=50)
plt.xlim(-1, 3)
plt.xticks([0, 1, 2, 3], fontsize=46)
plt.ylim(-3, 4)
plt.yticks([-2, 0, 2, 4], fontsize=46)
plt.grid(False)
plt.legend(fontsize=36, loc='upper left')

output_path_scatter = '/home/bioinfor6/BGM/fenziduiqi/5zbq/5zbq_mut_FEP.png'
plt.savefig(output_path_scatter, dpi=900)
plt.show()
