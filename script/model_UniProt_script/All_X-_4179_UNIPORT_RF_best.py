import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# 路径设置
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
output_dir = '/home/data1/BGM/fenziduiqi/All_X-_UNIPORT_4179_RF_best_result'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV 文件
df = pd.read_csv(csv_file)

# 获取 uniprot_ids
uniprot_ids = df['uniprot_ids'].values

# 解析嵌入向量
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

# 处理 X1, X2, ECFP 列
X1 = np.array(df['X1'].apply(parse_emb).tolist())
X2 = np.array(df['X2'].apply(parse_emb).tolist())
df_ecfp = np.array(df['ECFP'].apply(parse_emb).tolist())
y = np.array(df['ddG'])

# 合并特征：(X1-X2, df_ecfp)
X_diff = X1 - X2  # 计算 X1 和 X2 的差值
X = np.concatenate((X_diff, df_ecfp), axis=1)

# 存储皮尔逊结果
pearson_results = []

# 最佳参数
n_estimators = 300
max_depth = None
min_samples_split = 5

# 进行 30 次迭代计算
for idx in range(30):
    # 使用不同的随机种子
    random_seed = random.randint(1, 9999999)
    print(f"Random seed: {random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # ====================== 基于 uniprot_ids 划分数据集 ======================
    unique_ids = np.unique(uniprot_ids)

    # 将 unique_ids 按照 80% (train + val) 和 20% test 划分
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=random_seed)
    
    # 将 temp_ids 再按照一半一半划分：10% val 和 10% test
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_seed)

    # 根据掩码获取对应行
    train_mask = np.isin(uniprot_ids, train_ids)
    val_mask = np.isin(uniprot_ids, val_ids)
    test_mask = np.isin(uniprot_ids, test_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # 验证划分比例
    total_size = len(X)
    print(f"Run {idx+1}/30")
    print(f"总样本数: {total_size}")
    print(f"训练集大小: {len(X_train)} ({len(X_train)/total_size:.2%})")
    print(f"验证集大小: {len(X_val)} ({len(X_val)/total_size:.2%})")
    print(f"测试集大小: {len(X_test)} ({len(X_test)/total_size:.2%})")
    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # 初始化 RF 模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 验证集评估
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"Validation MSE: {val_mse:.4f}")
    
    # 测试集评估
    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_test, y_test_pred)
    spearman_corr, _ = spearmanr(y_test, y_test_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test Pearson: {pearson_corr:.4f}")
    print(f"Test Spearman: {spearman_corr:.4f}")
    
    # 生成预测图
    plt.figure(figsize=(14, 12))
    # 设置图的四个边框为黑色并加粗
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    # 调整子图边距
    plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
    plt.scatter(y_test, y_test_pred, color='blue', label='Data points', s=400)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color='gray', linewidth=2, linestyle='dashed', label='Ideal'
    )
    plt.xlabel('True ΔΔG', fontsize=46)
    plt.ylabel('Predicted ΔΔG', fontsize=46)
    plt.title('True vs Predicted ΔΔG - RF', fontsize=46)
    plt.xlim(-5.5, 5)
    plt.xticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
    plt.ylim(-5.5, 5)
    plt.yticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
    plt.legend(loc='upper left', fontsize=42)
    plt.text(
        0.55, 0.12, f'Pearson: {pearson_corr:.2f}',
        transform=plt.gca().transAxes, fontsize=44
    )
    plt.text(
        0.55, 0.05, f'Spearman: {spearman_corr:.2f}',
        transform=plt.gca().transAxes, fontsize=44
    )
    
    # 保存预测图
    figname = f'RF_prediction_run{idx+1}.png'
    output_path = os.path.join(output_dir, figname)
    plt.savefig(output_path, dpi=900)
    plt.close()
    
    # 保存预测结果
    prediction_file = f'RF_prediction_run{idx+1}.csv'
    csv_output_path = os.path.join(output_dir, prediction_file)
    results_df = pd.DataFrame({'True DDG': y_test, 'Predicted DDG': y_test_pred})
    results_df.to_csv(csv_output_path, index=False)
    
    # 记录皮尔逊结果
    pearson_results.append((idx+1, mae, rmse, pearson_corr, spearman_corr))
    
    # 删除模型以释放内存
    del model

# 保存皮尔逊结果
results_df = pd.DataFrame(
    pearson_results,
    columns=['Run', 'MAE', 'RMSE', 'Pearson Corr', 'Spearman']
)
results_df.to_csv(os.path.join(output_dir, 'RF_UNIPORT_best_results.csv'), index=False)