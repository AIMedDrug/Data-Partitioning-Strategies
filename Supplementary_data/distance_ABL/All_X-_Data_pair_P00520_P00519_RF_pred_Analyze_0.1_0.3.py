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
root_output_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pair_P00520_p00519_RF_pred_Analyze_0.1_0.3_result'
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
    '5DC4': 4, '2GQG': 20, '2V7A': 8, '6AMW': 3,
    '6XR6': 1, '4J9H': 1, '4JJB': 1, '7N9G': 1
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

# ====== 剩余为query（初始 127 条） ======
anchor_indices = set(anchor_df.index)
base_query_df = test_df.loc[~test_df.index.isin(anchor_indices)].copy()
print(f"Anchor总数: {len(anchor_df)}, 初始Query总数: {len(base_query_df)}")
assert len(anchor_df) == 39
assert len(base_query_df) == 127

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
            d1, d2 = group.loc[i], group.loc[j]
            x1 = np.concatenate([parse_emb(d1[x1_col]) - parse_emb(d1[x2_col]), parse_emb(d1[ecfp_col])])
            x2 = np.concatenate([parse_emb(d2[x1_col]) - parse_emb(d2[x2_col]), parse_emb(d2[ecfp_col])])
            X_pair = np.stack([x1, x2], axis=0)
            y_pair = d1[ddg_col] - d2[ddg_col]
            pairs_X.append(X_pair); pairs_y.append(y_pair); pairs_info.append((i, j, uniprot))
    return pairs_X, pairs_y, pairs_info

def make_cross_pairs(df_anchor, df_query, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X, pairs_y, pairs_info = [], [], []
    for j, i in product(df_query.index, df_anchor.index):
        d_query, d_anchor = df_query.loc[j], df_anchor.loc[i]
        x_query  = np.concatenate([parse_emb(d_query[x1_col]) - parse_emb(d_query[x2_col]), parse_emb(d_query[ecfp_col])])
        x_anchor = np.concatenate([parse_emb(d_anchor[x1_col]) - parse_emb(d_anchor[x2_col]), parse_emb(d_anchor[ecfp_col])])
        X_pair = np.stack([x_query, x_anchor], axis=0)
        y_pair = d_query[ddg_col] - d_anchor[ddg_col]
        pairs_X.append(X_pair); pairs_y.append(y_pair); pairs_info.append((j, i, d_query[uniprot_col]))
    return pairs_X, pairs_y, pairs_info

def df_all_text(df): return df.astype(str)

def prepare_data_for_rf(pairs_X):
    if len(pairs_X) == 0: return np.array([])
    feature_dim = pairs_X[0].shape[1]
    return np.array([np.concatenate([p[0], p[1]]) for p in pairs_X])

# ====== 实验比例划分 (10% 和 30%) ======
ratios = {"ratio0.1": 0.1, "ratio0.3": 0.3}
x1_col, x2_col, ecfp_col, ddg_col, uniprot_col = 'x1','x2','ECFP','ddG','UNIPROT'

for ratio_name, ratio_val in ratios.items():
    print(f"\n========== 当前划分: {ratio_name} ==========")
    sub_output_dir = os.path.join(root_output_dir, ratio_name)
    os.makedirs(sub_output_dir, exist_ok=True)

    # 随机抽取部分 query 样本合入 train
    sampled_query = base_query_df.sample(frac=ratio_val, random_state=2024)
    remain_query  = base_query_df.drop(sampled_query.index)

    # train = 原 train + anchor + sampled_query
    combined_train_df = pd.concat([train_df, anchor_df, sampled_query], axis=0)
    query_df = remain_query.copy()

    print(f"Train set: {len(combined_train_df)}, Query set: {len(query_df)} (抽取 {len(sampled_query)})")

    # ====== 配对 ======
    tX, ty, tinfo   = make_pairs(combined_train_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)
    aqX, aqy, aqinfo = make_cross_pairs(anchor_df, query_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)

    # 保存配对信息
    pair_info_dir = os.path.join(sub_output_dir, 'pair_info')
    os.makedirs(pair_info_dir, exist_ok=True)
    df_all_text(pd.DataFrame(tinfo, columns=['idx1','idx2','UNIPROT'])).to_csv(
        os.path.join(pair_info_dir, 'train_pairs_info.csv'), index=False)
    df_all_text(pd.DataFrame(aqinfo, columns=['idx_query','idx_anchor','UNIPROT'])).to_csv(
        os.path.join(pair_info_dir, 'query_anchor_pairs_info.csv'), index=False)

    # ====== 模型循环 ======
    all_results_rows = []
    for repeat_idx in range(30):
        print(f"---- 第 {repeat_idx+1} 次模型训练 ----")
        output_dir = os.path.join(sub_output_dir, f'result_{repeat_idx+1}')
        os.makedirs(output_dir, exist_ok=True)

        X_train = prepare_data_for_rf(tX).astype(np.float32)
        y_train = np.array(ty, dtype=np.float32)
        X_test  = prepare_data_for_rf(aqX).astype(np.float32)
        y_test  = np.array(aqy, dtype=np.float32)

        rf_model = cuRF(
            n_estimators=100, max_depth=20,
            random_state=repeat_idx, n_streams=8
        )
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        if hasattr(y_pred, "get"): y_pred = y_pred.get()

        # 还原 query ΔΔG
        query_count, anchor_count = len(query_df), len(anchor_df)
        y_pred_matrix = y_pred.reshape((query_count, anchor_count))
        anchor_ddG = anchor_df['ddG'].values
        q_pred_mean = (y_pred_matrix + anchor_ddG).mean(axis=1)

        # 真值
        q_real = query_df['ddG'].values

        # 评估
        pearson_corr, _  = pearsonr(q_real, q_pred_mean)
        spearman_corr, _ = spearmanr(q_real, q_pred_mean)

        # 保存每个 query 的结果
        per_query_df = pd.DataFrame({
            'PDB'               : query_df['PDB'].values,
            'Mutation'          : query_df['Mutation'].values,
            'uniprot_ids'       : query_df['UNIPROT'].values,
            'ddG'               : q_real,
            'Pred_ddG'          : q_pred_mean,
            'Category_By_Distance': query_df['Category_By_Distance'].values,
            'Repeat_Idx'        : [repeat_idx+1] * query_count,
            'Spearman_All'      : [spearman_corr] * query_count
        })
        all_results_rows.append(per_query_df)
        per_query_df.to_csv(os.path.join(output_dir, 'RF_spearman_results.csv'), index=False)

        # 分类 spearman
        cat_spearman = {}
        for cat_name, sub_df in per_query_df.groupby('Category_By_Distance'):
            if len(sub_df) > 1:
                s_corr, _ = spearmanr(sub_df['ddG'], sub_df['Pred_ddG'])
            else:
                s_corr = np.nan
            cat_spearman[cat_name] = s_corr

        # summary 行
        summary_row = {
            'Repeat_Idx': repeat_idx+1,
            'Spearman_All': spearman_corr,
            'Spearman_binding pocket': cat_spearman.get('binding pocket', np.nan),
            'Spearman_distal allosteric': cat_spearman.get('distal allosteric', np.nan),
            'Spearman_intermediate': cat_spearman.get('intermediate', np.nan)
        }

        summary_out_path = os.path.join(sub_output_dir, 'RF_spearman_summary_by_category.csv')
        if not os.path.exists(summary_out_path):
            pd.DataFrame([summary_row]).to_csv(summary_out_path, index=False)
        else:
            pd.DataFrame([summary_row]).to_csv(summary_out_path, mode='a', header=False, index=False)

        # ====== 绘制散点图 ======
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        for spine in ax.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(2)
        plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
        plt.scatter(q_real, q_pred_mean, color='blue', label='Data points', s=400)
        plt.plot([min(q_real), max(q_real)], [min(q_real), max(q_real)], 'gray', linewidth=2, linestyle='dashed', label='Ideal')
        plt.xlabel('True ΔΔG (Query)', fontsize=46)
        plt.ylabel('Predicted ΔΔG (Query)', fontsize=46)
        plt.title(f'Query True vs Predicted ΔΔG - RF ({ratio_name})', fontsize=46)
        plt.xlim(-5.5, 5); plt.ylim(-5.5, 5)
        plt.xticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
        plt.yticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
        plt.legend(loc='upper left', fontsize=42)
        plt.text(0.55, 0.12, f'Pearson: {pearson_corr:.2f}', transform=ax.transAxes, fontsize=44)
        plt.text(0.55, 0.05, f'Spearman: {spearman_corr:.2f}', transform=ax.transAxes, fontsize=44)
        plt.savefig(os.path.join(output_dir, 'RF_query_true_vs_pred.png'), dpi=900)
        plt.close()

    # 汇总 30 次
    if len(all_results_rows) > 0:
        final_df = pd.concat(all_results_rows, ignore_index=True)
        final_df.to_csv(os.path.join(sub_output_dir, 'RF_spearman_results_all.csv'), index=False)
        print(f"{ratio_name} 的所有 Spearman 已保存。")
