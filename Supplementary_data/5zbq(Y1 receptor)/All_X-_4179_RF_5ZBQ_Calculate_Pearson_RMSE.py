import pandas as pd

# 文件路径
csv_file = '/home/bioinfor6/BGM/fenziduiqi/5zbq/RF_pred_result/test_pairs_pearson_rmse_summary.csv'

# 读取数据
df = pd.read_csv(csv_file)

# 计算均值和标准差
pearson_mean = df['Pearson'].mean()
pearson_std = df['Pearson'].std()
rmse_mean = df['RMSE'].mean()
rmse_std = df['RMSE'].std()

# 按“均值 ± SD”格式输出
print(f"Pearson: {pearson_mean:.4f} ± {pearson_std:.4f}")
print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
