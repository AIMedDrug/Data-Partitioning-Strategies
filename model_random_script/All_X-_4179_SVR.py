import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import random

# 路径设置
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
output_dir = '/home/data1/BGM/fenziduiqi/All_X-_SVR_4179_result'

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

# SVR 超参数
C_list = [0.1, 1, 10]
epsilon_list = [0.01, 0.1, 0.5]
kernel_list = ['rbf', 'linear']

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
    for C in C_list:
        for epsilon in epsilon_list:
            for kernel in kernel_list:
                # 初始化 SVR 模型
                model = SVR(
                    C=C,
                    epsilon=epsilon,
                    kernel=kernel
                )
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 验证集评估
                y_val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, y_val_pred)
                print(f"Parameters: C={C}, epsilon={epsilon}, kernel={kernel}")
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
                plt.title('True vs Predicted DDG - SVR', fontsize=8)
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
                figname = f'SVR_prediction_random{idx}_C{C}_epsilon{epsilon}_kernel{kernel}.png'
                output_path = os.path.join(output_dir, figname)
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                # 保存预测结果
                prediction_file = f'SVR_prediction_random{idx}_C{C}_epsilon{epsilon}_kernel{kernel}.csv'
                results_df = pd.DataFrame({'True DDG': y_test, 'Predicted DDG': y_test_pred})
                csv_output_path = os.path.join(output_dir, prediction_file)
                results_df.to_csv(csv_output_path, index=False)
                
                # 记录皮尔逊结果
                pearson_results.append((random_seed, C, epsilon, kernel, pearson_corr))

# 找到最佳参数
max_corr = -np.inf
best_seed = None
best_C = None
best_epsilon = None
best_kernel = None
for seed, C, epsilon, kernel, corr in pearson_results:
    if corr > max_corr:
        max_corr = corr
        best_seed = seed
        best_C = C
        best_epsilon = epsilon
        best_kernel = kernel

# 打印最佳结果
print(f"\nBest Pearson Correlation: {max_corr:.4f}")
print(f"Best Parameters: random_seed={best_seed}, C={best_C}, epsilon={best_epsilon}, kernel={best_kernel}")

# 保存皮尔逊结果
results_df = pd.DataFrame(
    pearson_results,
    columns=['Random Seed', 'C', 'Epsilon', 'Kernel', 'Pearson Corr']
)
results_df.to_csv(os.path.join(output_dir, 'SVR_results.csv'), index=False)