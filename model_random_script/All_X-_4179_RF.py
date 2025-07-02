import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import random

# 路径设置
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
output_dir = '/home/data1/BGM/fenziduiqi/All_X-_RF_4179_result'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV 文件
df = pd.read_csv(csv_file)

# 解析嵌入向量
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

# 处理 X1, X2, ECFP 列
X1 = np.array(df['X1'].apply(parse_emb).tolist())
X2 = np.array(df['X2'].apply(parse_emb).tolist())
df_ecfp = np.array(df['ECFP'].apply(parse_emb).tolist())
y = np.array(df['ddG'])

# 合并特征
X = np.concatenate((X1 - X2, df_ecfp), axis=1)

# 存储皮尔逊结果
pearson_results = []

# RF 超参数
n_estimators_list = [100, 200, 300]
max_depth_list = [10, 20, None]
min_samples_split_list = [2, 5, 10]

# 10 次随机划分
for idx in range(10):
    random_seed = random.randint(1, 9999999)
    print(f"Random seed: {random_seed}")
    
    # 数据划分：80% 训练，10% 验证，10% 测试
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # 超参数调优
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
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
                print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
                print(f"Validation MSE: {val_mse:.4f}")
                
                # 测试集评估
                y_test_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_test_pred)
                pearson_corr, _ = pearsonr(y_test, y_test_pred)
                spearman_corr, _ = spearmanr(y_test, y_test_pred)
                
                print(f"Test MSE: {mse:.4f}")
                print(f"Test Pearson: {pearson_corr:.4f}")
                print(f"Test Spearman: {spearman_corr:.4f}")
                
                # 生成预测图
                plt.figure(figsize=(5, 4))
                plt.scatter(y_test, y_test_pred, color='blue', label='Data points', s=30)
                plt.plot(
                    [min(y_test), max(y_test)],
                    [min(y_test), max(y_test)],
                    color='red', linewidth=2, label='Ideal'
                )
                plt.xlabel('True DDG', fontsize=8)
                plt.ylabel('Predicted DDG', fontsize=8)
                plt.title('True vs Predicted DDG - RF', fontsize=8)
                plt.legend(loc='upper left', fontsize=8)
                plt.text(
                    0.05, 0.83, f'Pearson Corr: {pearson_corr:.2f}',
                    transform=plt.gca().transAxes, fontsize=8
                )
                plt.text(
                    0.05, 0.78, f'Spearman Corr: {spearman_corr:.2f}',
                    transform=plt.gca().transAxes, fontsize=8
                )
                
                # 保存预测图
                figname = f'RF_prediction_random{idx}_estimators{n_estimators}_depth{max_depth}_split{min_samples_split}.png'
                output_path = os.path.join(output_dir, figname)
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                # 保存预测结果
                prediction_file = f'RF_prediction_random{idx}_estimators{n_estimators}_depth{max_depth}_split{min_samples_split}.csv'
                csv_output_path = os.path.join(output_dir, prediction_file)
                results_df = pd.DataFrame({'True DDG': y_test, 'Predicted DDG': y_test_pred})
                results_df.to_csv(csv_output_path, index=False)
                
                # 记录皮尔逊结果，包含 random_seed
                pearson_results.append((random_seed, n_estimators, max_depth, min_samples_split, pearson_corr))

# 找到最佳参数
max_corr = -np.inf
best_seed = None
best_estimators = None
best_depth = None
best_split = None
for seed, estimators, depth, split, corr in pearson_results:
    if corr > max_corr:
        max_corr = corr
        best_seed = seed
        best_estimators = estimators
        best_depth = depth
        best_split = split

# 打印最佳结果
print(f"\nBest Pearson Correlation: {max_corr:.4f}")
print(f"Best Parameters: random_seed={best_seed}, n_estimators={best_estimators}, max_depth={best_depth}, min_samples_split={best_split}")

# 保存皮尔逊结果
results_df = pd.DataFrame(
    pearson_results,
    columns=['Random Seed', 'N Estimators', 'Max Depth', 'Min Samples Split', 'Pearson Corr']
)
results_df.to_csv(os.path.join(output_dir, 'RF_results.csv'), index=False)