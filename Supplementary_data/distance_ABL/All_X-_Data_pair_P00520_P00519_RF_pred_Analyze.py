import os
import pandas as pd
import numpy as np
import random
from itertools import combinations, product
from cuml.ensemble import RandomForestRegressor as cuRF
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ====== 路径设置 ======
csv_file = '/home/bioinfor6/BGM/fenziduiqi/MdrDB_mutation_embed_P00520_P00519.csv'
root_output_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pair_P00520_p00519_RF_pred_Analyze_result'
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

# ====== 配对函数 ======
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
    return df.astype(str)

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

# 输出一次数据配对信息
output_info_dir = os.path.join(root_output_dir, 'pair_info')
os.makedirs(output_info_dir, exist_ok=True)
df_all_text(pd.DataFrame(tinfo, columns=['idx1','idx2','UNIPROT'])).to_csv(
    os.path.join(output_info_dir, 'train_pairs_info.csv'), index=False)
df_all_text(pd.DataFrame(aqinfo, columns=['idx_query','idx_anchor','UNIPROT'])).to_csv(
    os.path.join(output_info_dir, 'query_anchor_pairs_info.csv'), index=False)
print("所有配对数据已保存到 CSV。")

# =========== 模型部分循环 30 次 ===========
spearman_summary = []   # Spearman 汇总表
all_results = []        # 保存所有 query 的预测结果

for repeat_idx in range(30):
    print(f"\n======= 第 {repeat_idx + 1} 次模型训练 =======")
    output_dir = os.path.join(root_output_dir, f'result_{repeat_idx+1}')
    os.makedirs(output_dir, exist_ok=True)

    X_train = prepare_data_for_rf(tX).astype(np.float32)
    y_train = np.array(ty, dtype=np.float32)
    X_test  = prepare_data_for_rf(aqX).astype(np.float32)
    y_test  = np.array(aqy, dtype=np.float32)

    rf_model = cuRF(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        random_state=repeat_idx,
        n_streams=8
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    if hasattr(y_pred, "get"):
        y_pred = y_pred.get()

    # ====== 还原 query ΔΔG ======
    query_count  = len(query_df)
    anchor_count = len(anchor_df)
    y_pred_matrix = y_pred.reshape((query_count, anchor_count))
    anchor_ddG     = anchor_df['ddG'].values
    q_pred_matrix = y_pred_matrix + anchor_ddG
    q_pred_mean_per_query = q_pred_matrix.mean(axis=1)

    # ====== 真实值 & 分类标签 ======
    q_real = query_df['ddG'].values
    query_regions = query_df["Category_By_Distance"].values

    # ====== Spearman 计算 ======
    spearman_results = {"repeat_idx": repeat_idx+1}
    spearman_results["All"] = spearmanr(q_real, q_pred_mean_per_query)[0]
    for region_name in ["binding pocket", "distal allosteric", "intermediate"]:
        mask = (query_regions == region_name)
        if mask.sum() > 1:
            spearman_val, _ = spearmanr(q_real[mask], q_pred_mean_per_query[mask])
        else:
            spearman_val = np.nan
        spearman_results[region_name] = spearman_val
    spearman_summary.append(spearman_results)

    # ====== 保存到总表（新增部分） ======
    temp_df = pd.DataFrame({
        'PDB': query_df['PDB'].values,
        'Mutation': query_df['Mutation'].values,
        'uniprot_ids': query_df['UNIPROT'].values,
        'ddG': q_real,
        'Pred_ddG': q_pred_mean_per_query,
        'Repeat_Idx': repeat_idx+1,
        'Spearman_Corr': spearman_results["All"]
    })
    all_results.append(temp_df)

    # ====== 绘制散点图 ======
    q_pred = q_pred_mean_per_query
    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
    plt.scatter(q_real, q_pred, color='blue', label='Data points', s=400)
    plt.plot([min(q_real), max(q_real)], [min(q_real), max(q_real)],
             color='gray', linewidth=2, linestyle='dashed', label='Ideal')
    plt.xlabel('True ΔΔG (Query)', fontsize=46)
    plt.ylabel('Predicted ΔΔG (Query)', fontsize=46)
    plt.title('Query True vs Predicted ΔΔG - RF', fontsize=46)
    plt.xlim(-5.5, 5)
    plt.ylim(-5.5, 5)
    plt.legend(loc='upper left', fontsize=42)
    plt.close()

# ====== 保存 Spearman 汇总表 ======
summary_df = pd.DataFrame(spearman_summary)
summary_csv_path = os.path.join(root_output_dir, 'query_pred_ddG_means_spearman_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)

# ====== 保存总表（所有 query 的预测结果） ======
all_results_df = pd.concat(all_results, ignore_index=True)
all_results_csv = os.path.join(root_output_dir, 'query_pred_ddG_all_results.csv')
all_results_df.to_csv(all_results_csv, index=False)

print(f"所有30次 Spearman 已保存到：{summary_csv_path}")
print(f"所有 Query 的预测结果已保存到：{all_results_csv}")
