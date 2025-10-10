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
csv_file = '/home/bioinfor6/BGM/fenziduiqi/MdrDB_mutation_embed_P00533.csv'
output_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_RF_P00533_random_result'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV 文件
df = pd.read_csv(csv_file)

# 解析嵌入向量
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

# spearman 结果记录（最终大表）
all_results = []

# RF 固定参数
n_estimators = 100
max_depth = 20
min_samples_split = 2

# 运行多次随机种子实验
for idx in range(30):
    random_number = random.randint(1, 9999999)
    print(f"Random seed: {random_number}")

    # 数据集划分
    test_df = df[df['uniprot_ids'] == 'P00533']
    train_df = df[df['uniprot_ids'] != 'P00533']

    print(f"Test set size (P00533): {len(test_df)}")
    print(f"Train set size: {len(train_df)}")

    # 处理训练数据
    X1 = np.array(train_df['X1'].apply(parse_emb).tolist())
    X2 = np.array(train_df['X2'].apply(parse_emb).tolist())
    df_ecfp = np.array(train_df['ECFP'].apply(parse_emb).tolist())
    X_train = np.concatenate((X1 - X2, df_ecfp), axis=1)
    y_train = np.array(train_df['ddG'])

    # 划分训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_number)

    # 处理测试数据
    X1_test = np.array(test_df['X1'].apply(parse_emb).tolist())
    X2_test = np.array(test_df['X2'].apply(parse_emb).tolist())
    df_ecfp_test = np.array(test_df['ECFP'].apply(parse_emb).tolist())
    X_test = np.concatenate((X1_test - X2_test, df_ecfp_test), axis=1)
    y_test = np.array(test_df['ddG'])

    print(f"Train set 大小: {len(y_train)}")
    print(f"Test set 大小: {len(y_test)}")
    print(f"check the shape: X_train={X_train.shape}, X_test={X_test.shape}")

    # 初始化并训练 RF 模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
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
    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
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
    figname = f'RF_prediction_random{idx}.png'
    output_path = os.path.join(output_dir, figname)
    plt.savefig(output_path, dpi=900)
    plt.close()

    # 保存单次预测结果（原始功能保留）
    prediction_file = f'RF_prediction_random{idx}.csv'
    csv_output_path = os.path.join(output_dir, prediction_file)
    results_df = pd.DataFrame({'True DDG': y_test, 'Predicted DDG': y_test_pred})
    results_df.to_csv(csv_output_path, index=False)

    # 保存到总表中（附带 PDB, Mutation, uniprot_ids）
    temp_df = pd.DataFrame({
        'PDB': test_df['PDB'].values,
        'Mutation': test_df['Mutation'].values,
        'uniprot_ids': test_df['uniprot_ids'].values,
        'ddG': y_test,
        'Pred_ddG': y_test_pred,
        'Random_Number': random_number,
        'Spearman_Corr': spearman_corr
    })
    all_results.append(temp_df)

# 合并所有实验结果
final_df = pd.concat(all_results, ignore_index=True)

# 保存为最终结果文件
output_file = os.path.join(output_dir, 'RF_spearman_results.csv')
final_df.to_csv(output_file, index=False)

print(f"\n结果已保存到 {output_file}")
