import os
import pandas as pd
import numpy as np
import random
from itertools import combinations, product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ====== 路径设置 ======
csv_file = '/home/bioinfor6/BGM/fenziduiqi/check_embed_list.csv'
root_output_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pairing_UNIPROT_P00520_p00519_RF_pred_result'
os.makedirs(root_output_dir, exist_ok=True)

# ====== 加载 CSV 文件 ======
df = pd.read_csv(csv_file)

if 'UNIPROT' not in df.columns:
    df['UNIPROT'] = df.iloc[:, 6]
    df['ddG']     = df.iloc[:, 5]
    df['x1']      = df.iloc[:, 7]
    df['x2']      = df.iloc[:, 8]
    df['ECFP']    = df.iloc[:, 9]

# ====== anchor目标PDB及数量 ======
target_pdb_counts = {
    '5DC4': 4,
    '2GQG': 20,
    '2V7A': 8,
    '6AMW': 3,
    '6XR6': 1,
    '4J9H': 1,
    '4JJB': 1,
    '7N9G': 1
}
total_anchor = sum(target_pdb_counts.values())

# ====== 按UNIPROT筛选test/train ======
test_df  = df[df['UNIPROT'].isin(['P00520', 'P00519'])].copy()
train_df = df[~df['UNIPROT'].isin(['P00520', 'P00519'])].copy()

# ====== 精准筛选anchor ======
anchor_rows = []
for pdb_id, need_count in target_pdb_counts.items():
    subset = test_df[test_df['PDB'] == pdb_id]
    if len(subset) < need_count:
        raise ValueError(f"PDB ID {pdb_id} only has {len(subset)} records, need {need_count}!")
    anchor_rows.append(subset.sample(n=need_count, random_state=2024))  # 固定随机种子
anchor_df = pd.concat(anchor_rows)
anchor_df = anchor_df.sample(frac=1, random_state=2024)  # 打乱

# ====== 剩余为query ======
anchor_indices = set(anchor_df.index)
query_df = test_df.loc[~test_df.index.isin(anchor_indices)].copy()
print(f"Anchor总数: {len(anchor_df)}, Query总数: {len(query_df)}")
assert len(anchor_df) == 39
assert len(query_df) == 127

# ====== anchor放入train ======
combined_train_df = pd.concat([train_df, anchor_df], axis=0)

print(f"Train set (train+anchor): {len(combined_train_df)}")
print(f"Test anchor: {len(anchor_df)}, Test query: {len(query_df)}")

x1_col, x2_col, ecfp_col, ddg_col, uniprot_col = 'x1', 'x2', 'ECFP', 'ddG', 'UNIPROT'

# ====== 配对 ======
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

def make_pairs(df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X, pairs_y, pairs_info = [], [], []
    grouped = df.groupby(uniprot_col)
    for uniprot, group in grouped:
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        for i, j in combinations(idxs, 2):
            d1 = group.loc[i]
            d2 = group.loc[j]
            x1 = np.concatenate([parse_emb(d1[x1_col]) - parse_emb(d1[x2_col]), parse_emb(d1[ecfp_col])])
            x2 = np.concatenate([parse_emb(d2[x1_col]) - parse_emb(d2[x2_col]), parse_emb(d2[ecfp_col])])
            X_pair = np.stack([x1, x2], axis=0)
            y_pair = d1[ddg_col] - d2[ddg_col]
            pairs_X.append(X_pair)
            pairs_y.append(y_pair)
            pairs_info.append((i, j, uniprot))
    return pairs_X, pairs_y, pairs_info

def make_cross_pairs(df_anchor, df_query, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X, pairs_y, pairs_info = [], [], []
    query_indices  = df_query.index.tolist()
    anchor_indices = df_anchor.index.tolist()
    for j, i in product(query_indices, anchor_indices):
        d_query  = df_query.loc[j]
        d_anchor = df_anchor.loc[i]
        x_query  = np.concatenate([parse_emb(d_query[x1_col]) - parse_emb(d_query[x2_col]), parse_emb(d_query[ecfp_col])])
        x_anchor = np.concatenate([parse_emb(d_anchor[x1_col]) - parse_emb(d_anchor[x2_col]), parse_emb(d_anchor[ecfp_col])])
        X_pair = np.stack([x_query, x_anchor], axis=0)
        y_pair = d_query[ddg_col] - d_anchor[ddg_col]
        pairs_X.append(X_pair)
        pairs_y.append(y_pair)
        pairs_info.append((j, i, d_query[uniprot_col]))
    return pairs_X, pairs_y, pairs_info

def df_all_text(df):
    return df.applymap(str)

def prepare_data_for_rf(pairs_X):
    n_pairs = len(pairs_X)
    if n_pairs == 0:
        return np.array([])
    feature_dim = pairs_X[0].shape[1]
    X_flat = np.zeros((n_pairs, 2 * feature_dim))
    for idx, pair in enumerate(pairs_X):
        X_flat[idx] = np.concatenate([pair[0], pair[1]])
    return X_flat

# ====== 开始配对 ======
tX, ty, tinfo = make_pairs(combined_train_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)
aqX, aqy, aqinfo = make_cross_pairs(anchor_df, query_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)

# 输出一次数据配对信息，全部循环共享
output_info_dir = os.path.join(root_output_dir, 'pair_info')
os.makedirs(output_info_dir, exist_ok=True)
df_all_text(pd.DataFrame(tinfo, columns=['idx1','idx2','UNIPROT'])).to_csv(
    os.path.join(output_info_dir, 'train_pairs_info.csv'), index=False)
df_all_text(pd.DataFrame(aqinfo, columns=['idx_query','idx_anchor','UNIPROT'])).to_csv(
    os.path.join(output_info_dir, 'query_anchor_pairs_info.csv'), index=False)
print("所有配对数据已保存到 CSV。")

# =========== 模型部分循环 30 次 ===========
pearson_rmse_summary = []  # 汇总表

for repeat_idx in range(30):
    print(f"\n======= 第 {repeat_idx + 1} 次模型训练 =======")
    output_dir = os.path.join(root_output_dir, f'result_{repeat_idx+1}')
    os.makedirs(output_dir, exist_ok=True)

    # ====== RF 模型训练与预测 ======
    print("\n开始 RF 模型训练和预测...")

    X_train = prepare_data_for_rf(tX)
    y_train = np.array(ty)
    X_test  = prepare_data_for_rf(aqX)
    y_test  = np.array(aqy)

    print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")

    # 随机森林超参数（仅 random_state 每次不同）
    n_estimators       = 100
    max_depth          = 20
    min_samples_split  = 2

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=repeat_idx,  # 用循环索引做随机种子
        n_jobs=-1
    )

    print("训练 RF 模型...")
    rf_model.fit(X_train, y_train)
    print("模型训练完成。")

    print("在测试集上进行预测...")
    y_pred = rf_model.predict(X_test)

    # 计算各项评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    # ====== 保存 Query-Anchor 详细预测结果 ======
    aqinfo_arr     = np.array(aqinfo)
    idx_query_arr  = aqinfo_arr[:, 0].astype(int)
    idx_anchor_arr = aqinfo_arr[:, 1].astype(int)

    query_ddG_arr  = query_df.loc[idx_query_arr,  'ddG'].values
    anchor_ddG_arr = anchor_df.loc[idx_anchor_arr, 'ddG'].values
    real_diff_arr  = query_ddG_arr - anchor_ddG_arr
    y_pred_arr     = y_pred

    query_anchor_detail_df = pd.DataFrame({
        'idx_query'       : idx_query_arr,
        'idx_anchor'      : idx_anchor_arr,
        'query_ddG'       : query_ddG_arr,
        'anchor_ddG'      : anchor_ddG_arr,
        'real_ddG_diff'   : real_diff_arr,
        'pred_ddG_diff'   : y_pred_arr
    }).applymap(str)
    detail_csv_path = os.path.join(output_dir, "query_anchor_ddG_pairs_detail.csv")
    query_anchor_detail_df.to_csv(detail_csv_path, index=False)

    print("配对细节数据已保存到 query_anchor_ddG_pairs_detail.csv。")

    # ====== 追加 sum_ddG 和 分组均值，并合并 PDB ID ======
    pdbid_col = 'PDB'
    detail_df = pd.read_csv(detail_csv_path, dtype=str)

    detail_df['anchor_ddG'] = detail_df['anchor_ddG'].astype(float)
    detail_df['pred_ddG_diff'] = detail_df['pred_ddG_diff'].astype(float)
    detail_df['sum_ddG'] = detail_df['anchor_ddG'] + detail_df['pred_ddG_diff']
    mean_sum_ddG = detail_df.groupby('idx_query')['sum_ddG'].mean().reset_index()
    mean_sum_ddG.rename(columns={'sum_ddG': 'mean_sum_ddG_by_query'}, inplace=True)
    detail_df = detail_df.merge(mean_sum_ddG, on='idx_query', how='left')
    idx_query_to_pdbid = query_df[pdbid_col].to_dict()
    detail_df['query_PDB_ID'] = detail_df['idx_query'].map(idx_query_to_pdbid)
    cols = detail_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('query_PDB_ID')))
    detail_df = detail_df[cols]
    detail_df = detail_df.applymap(str)

    final_detail_path = os.path.join(output_dir, 'query_anchor_ddG_pairs_detail_with_means.csv')
    detail_df.to_csv(final_detail_path, index=False)
    print(f"sum_ddG、分组均值和 query_PDB_ID 已保存到 {final_detail_path}。")

    # ====== 批量还原并统计每个 Query 的预测 ΔΔG ======
    print("\n【开始评估每个 Query 的预测 ΔΔG 值】")
    query_count  = len(query_df)
    anchor_count = len(anchor_df)
    y_pred_matrix = y_pred.reshape((query_count, anchor_count))
    anchor_ddG     = anchor_df['ddG'].values
    q_pred_matrix = y_pred_matrix + anchor_ddG
    q_pred_mean_per_query = q_pred_matrix.mean(axis=1)
    overall_q_pred_mean   = q_pred_mean_per_query.mean()
    print(f"所有 Query 预测 ΔΔG 的整体均值: {overall_q_pred_mean:.4f}")

    tmp_df = pd.DataFrame({
        "query_idx"  : query_df.index.values,
        "query_PDB_ID": query_df[pdbid_col].values,
        "q_pred_mean": q_pred_mean_per_query,
        "q_real"     : query_df['ddG'].values
    }).applymap(str)
    tmp_df.to_csv(os.path.join(output_dir, "query_pred_ddG_means.csv"), index=False)

    # ====== 新增：读取query_pred_ddG_means.csv计算Pearson和RMSE，并存到汇总表 ======
    means_df = pd.read_csv(os.path.join(output_dir, "query_pred_ddG_means.csv"))
    q_real_vals = means_df["q_real"].astype(float).values
    q_pred_means = means_df["q_pred_mean"].astype(float).values
    pearson_means, _ = pearsonr(q_real_vals, q_pred_means)
    rmse_means = np.sqrt(mean_squared_error(q_real_vals, q_pred_means))
    print(f"【query_pred_ddG_means.csv】 Pearson: {pearson_means:.4f}, RMSE: {rmse_means:.4f}")
    pearson_rmse_summary.append({
        "repeat_idx": repeat_idx+1,
        "Pearson": pearson_means,
        "RMSE": rmse_means
    })

    # ====== 打印最终评估指标 ======
    print("\n模型评估结果：")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Pearson 相关系数: {pearson_corr:.4f}")
    print(f"  Spearman 相关系数: {spearman_corr:.4f}")

    y_test_mean  = np.mean(y_test)
    y_pred_mean  = np.mean(y_pred)
    print(f"\n  ddG 差值真实均值: {y_test_mean:.4f}")
    print(f"  ddG 差值预测均值: {y_pred_mean:.4f}")

    # 保存预测差值结果
    results_df = pd.DataFrame({
        'True_ddG_diff'      : y_test,
        'Predicted_ddG_diff' : y_pred
    }).applymap(str)
    results_df.to_csv(os.path.join(output_dir, 'RF_prediction_results.csv'), index=False)

    # 保存各项均值
    avg_df = pd.DataFrame({
        'Metric': [
            'True_ddG_diff_mean',
            'Predicted_ddG_diff_mean',
            'Overall_query_pred_ddG_mean'
        ],
        'Value': [
            y_test_mean,
            y_pred_mean,
            overall_q_pred_mean
        ]
    }).applymap(str)
    avg_df.to_csv(os.path.join(output_dir, 'RF_prediction_averages.csv'), index=False)

    # ====== 绘制：Query 真实 ΔΔG vs 预测 ΔΔG 均值 散点图 ======
    q_real = query_df['ddG'].values
    q_pred = q_pred_mean_per_query

    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
    plt.scatter(q_real, q_pred, color='blue', label='Data points', s=400)
    plt.plot(
        [min(q_real), max(q_real)],
        [min(q_real), max(q_real)],
        color='gray', linewidth=2, linestyle='dashed', label='Ideal'
    )
    plt.xlabel('True ΔΔG (Query)', fontsize=46)
    plt.ylabel('Predicted ΔΔG (Query)', fontsize=46)
    plt.title('Query True vs Predicted ΔΔG - RF', fontsize=46)
    plt.xlim(-5.5, 5)
    plt.xticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
    plt.ylim(-5.5, 5)
    plt.yticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
    plt.legend(loc='upper left', fontsize=42)
    pearson_q, _  = pearsonr(q_real, q_pred)
    spearman_q, _ = spearmanr(q_real, q_pred)
    plt.text(
        0.55, 0.12, f'Pearson: {pearson_q:.2f}',
        transform=plt.gca().transAxes, fontsize=44
    )
    plt.text(
        0.55, 0.05, f'Spearman: {spearman_q:.2f}',
        transform=plt.gca().transAxes, fontsize=44
    )

    figname    = f'RF_query_true_vs_pred_estimators{n_estimators}_depth{max_depth}_split{min_samples_split}_query_anchor.png'
    output_path = os.path.join(output_dir, figname)
    plt.savefig(output_path, dpi=900)
    plt.close()

    print(f"Query 真实-预测值散点图已保存到: {output_path}")
    print("本次模型预测和评估完成。")

# ====== 循环结束后保存所有30次的Pearson和RMSE汇总 ======
summary_df = pd.DataFrame(pearson_rmse_summary)
summary_csv_path = os.path.join(root_output_dir, 'query_pred_ddG_means_pearson_rmse_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"所有30次query_pred_ddG_means的Pearson和RMSE已保存到：{summary_csv_path}")
