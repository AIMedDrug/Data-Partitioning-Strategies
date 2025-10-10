import os
# 只曝光 GPU 1 给本进程（物理 GPU1 在进程内映射为逻辑 device 0）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import cupy as cp
import random
from itertools import combinations, product

# cuML: GPU 加速版随机森林
from cuml.ensemble import RandomForestRegressor as cuRF

# 指标仍用 sklearn/scipy（在 CPU 上计算）
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ====== 路径设置 ======
csv_file = '/home/bioinfor6/BGM/fenziduiqi/MdrDB_mutation_embed_P00533.csv'
root_output_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pairing_P00533_RF_result_0.1_0.3_Analyze'
os.makedirs(root_output_dir, exist_ok=True)

# ====== 加载 CSV 文件 ======
df = pd.read_csv(csv_file)

# 若缺省列名，按你之前的映射补齐
if 'UNIPROT' not in df.columns:
    df['UNIPROT'] = df.iloc[:, 6]
    df['ddG']     = df.iloc[:, 5]
    df['x1']      = df.iloc[:, 7]
    df['x2']      = df.iloc[:, 8]
    df['ECFP']    = df.iloc[:, 9]

# 统一一份便于输出的列名
if 'uniprot_ids' not in df.columns:
    df['uniprot_ids'] = df['UNIPROT']

# ====== 辅助函数 ======
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

def make_pairs(df_in, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    """同一 UNIPROT 内部做两两配对（训练用）"""
    pairs_X, pairs_y, pairs_info = [], [], []
    grouped = df_in.groupby(uniprot_col)
    for uniprot, group in grouped:
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        for i, j in combinations(idxs, 2):
            d1, d2 = group.loc[i], group.loc[j]
            x1 = np.concatenate([parse_emb(d1[x1_col]) - parse_emb(d1[x2_col]),
                                 parse_emb(d1[ecfp_col])])
            x2 = np.concatenate([parse_emb(d2[x1_col]) - parse_emb(d2[x2_col]),
                                 parse_emb(d2[ecfp_col])])
            X_pair = np.stack([x1, x2], axis=0)
            y_pair = d1[ddg_col] - d2[ddg_col]
            pairs_X.append(X_pair)
            pairs_y.append(y_pair)
            pairs_info.append((i, j, uniprot))
    return pairs_X, pairs_y, pairs_info

def make_cross_pairs(df_anchor, df_query, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    """Query 与 Anchor 交叉配对（测试用）"""
    pairs_X, pairs_y, pairs_info = [], [], []
    query_indices  = df_query.index.tolist()
    anchor_indices = df_anchor.index.tolist()
    for j, i in product(query_indices, anchor_indices):
        d_query  = df_query.loc[j]
        d_anchor = df_anchor.loc[i]
        x_query  = np.concatenate([parse_emb(d_query[x1_col]) - parse_emb(d_query[x2_col]),
                                   parse_emb(d_query[ecfp_col])])
        x_anchor = np.concatenate([parse_emb(d_anchor[x1_col]) - parse_emb(d_anchor[x2_col]),
                                   parse_emb(d_anchor[ecfp_col])])
        X_pair = np.stack([x_query, x_anchor], axis=0)
        y_pair = d_query[ddg_col] - d_anchor[ddg_col]
        pairs_X.append(X_pair)
        pairs_y.append(y_pair)
        pairs_info.append((j, i, d_query[uniprot_col]))
    return pairs_X, pairs_y, pairs_info

def df_all_text(df_in):
    return df_in.astype(str)

def prepare_data_for_rf(pairs_X):
    if len(pairs_X) == 0:
        return np.array([])
    n_pairs = len(pairs_X)
    feature_dim = pairs_X[0].shape[1]
    X_flat = np.zeros((n_pairs, 2 * feature_dim))
    for k, pair in enumerate(pairs_X):
        X_flat[k] = np.concatenate([pair[0], pair[1]])
    return X_flat

# ====== 参数与目标设定 ======
target_uniprot = 'P00533'
anchor_ratios  = [0.1, 0.3]

# ====== 主循环：遍历比例与种子 ======
for ratio in anchor_ratios:
    subdir = os.path.join(root_output_dir, f'ratio_{int(ratio*100)}')
    os.makedirs(subdir, exist_ok=True)
    print(f'\n===== Running ratio={ratio:.1f} ({subdir}) =====')

    # 汇总表
    all_results_rows = []

    for run_idx in range(30):
        random_number = random.randint(1, 9999999)
        print("随机种子：", random_number)

        # Test = P00533，Train = 其它
        test_df_all   = df[df['UNIPROT'] == target_uniprot].copy()
        train_df_base = df[df['UNIPROT'] != target_uniprot].copy()
        n_test_total  = len(test_df_all)

        if n_test_total == 0:
            print("⚠️ 没有找到 P00533 的样本，跳过。")
            continue

        # 抽取 anchor/query
        random.seed(random_number)
        np.random.seed(random_number)
        test_indices   = test_df_all.index.tolist()
        anchor_size    = int(n_test_total * ratio)
        anchor_indices = random.sample(test_indices, anchor_size) if anchor_size > 0 else []
        query_indices  = list(set(test_indices) - set(anchor_indices))

        anchor_df = test_df_all.loc[anchor_indices].copy()
        query_df  = test_df_all.loc[query_indices].copy()
        combined_train_df = pd.concat([train_df_base, anchor_df], axis=0)

        # 列名
        x1_col, x2_col, ecfp_col, ddg_col, uniprot_col = 'x1','x2','ECFP','ddG','UNIPROT'

        # 生成训练/测试配对
        tX, ty, tinfo    = make_pairs(combined_train_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)
        aqX, aqy, aqinfo = make_cross_pairs(anchor_df, query_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)

        run_out_dir = os.path.join(subdir, f'All_X-_Data_pairing_UNIPROT_{target_uniprot}_RF_pred_result_{ratio}_seed{random_number}')
        os.makedirs(run_out_dir, exist_ok=True)
        df_all_text(pd.DataFrame(tinfo,  columns=['idx1','idx2','UNIPROT'])).to_csv(
            os.path.join(run_out_dir, 'train_pairs_info.csv'), index=False)
        df_all_text(pd.DataFrame(aqinfo, columns=['idx_query','idx_anchor','UNIPROT'])).to_csv(
            os.path.join(run_out_dir, 'query_anchor_pairs_info.csv'), index=False)

        # ====== 准备 RF（将 numpy -> cupy 加速训练/预测）======
        X_train_np = prepare_data_for_rf(tX); y_train_np = np.array(ty)
        X_test_np  = prepare_data_for_rf(aqX); y_test_np = np.array(aqy)

        if X_train_np.size == 0 or X_test_np.size == 0:
            print("⚠️ 训练或测试数据为空，跳过本轮。")
            continue

        # 转为 GPU 上的 cupy 数组
        X_train = cp.asarray(X_train_np)
        y_train = cp.asarray(y_train_np)
        X_test  = cp.asarray(X_test_np)

        # cuML 随机森林（注意：cuML 不支持 min_samples_split 参数）
        rf_model = cuRF(
            n_estimators=100,
            max_depth=20,
            n_streams=8,       # 可根据 GPU 情况调整并行流
            random_state=42
        )

        # 训练（GPU）
        rf_model.fit(X_train, y_train)

        # 预测（GPU -> cupy）
        y_pred_cu = rf_model.predict(X_test)

        # 转回 CPU numpy 用于后续指标与还原
        y_pred = cp.asnumpy(y_pred_cu)

        query_count, anchor_count = len(query_df), len(anchor_df)
        if query_count == 0 or anchor_count == 0:
            print("⚠️ Query 或 Anchor 为空，跳过本轮。")
            continue

        # 还原每个 query 的 Pred_ddG
        y_pred_matrix = y_pred.reshape((query_count, anchor_count))
        anchor_ddG    = anchor_df['ddG'].values
        q_pred_matrix = y_pred_matrix + anchor_ddG
        q_pred_mean   = q_pred_matrix.mean(axis=1)
        q_real        = query_df['ddG'].values

        # 评估（CPU 计算）
        pearson_corr, _  = pearsonr(q_real, q_pred_mean)
        spearman_corr, _ = spearmanr(q_real, q_pred_mean)

        # 保存结果（增加 Category_By_Distance 信息）
        per_query_df = pd.DataFrame({
            'PDB'               : query_df['PDB'].values,
            'Mutation'          : query_df['Mutation'].values,
            'uniprot_ids'       : query_df['uniprot_ids'].values,
            'ddG'               : q_real,
            'Pred_ddG'          : q_pred_mean,
            'Category_By_Distance': query_df['Category_By_Distance'].values,
            'Random_Number'     : [random_number] * query_count,
            'Spearman_All'      : [spearman_corr] * query_count
        })
        all_results_rows.append(per_query_df)
        per_query_df.to_csv(os.path.join(run_out_dir, 'RF_spearman_results.csv'), index=False)

        # ====== 分类 spearman 统计 ======
        cat_spearman = {}
        for cat_name, sub_df in per_query_df.groupby('Category_By_Distance'):
            if len(sub_df) > 1:
                s_corr, _ = spearmanr(sub_df['ddG'], sub_df['Pred_ddG'])
            else:
                s_corr = np.nan
            cat_spearman[cat_name] = s_corr

        # 构造 summary 行（一个 run 一行）
        summary_row = {
            'Random_Number': random_number,
            'Spearman_All': spearman_corr,
            'Spearman_binding pocket': cat_spearman.get('binding pocket', np.nan),
            'Spearman_distal allosteric': cat_spearman.get('distal allosteric', np.nan),
            'Spearman_intermediate': cat_spearman.get('intermediate', np.nan)
        }

        summary_out_path = os.path.join(subdir, 'RF_spearman_summary_by_category.csv')
        if not os.path.exists(summary_out_path):
            pd.DataFrame([summary_row]).to_csv(summary_out_path, index=False)
        else:
            pd.DataFrame([summary_row]).to_csv(summary_out_path, mode='a', header=False, index=False)

        # 作图
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2)
        plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
        plt.scatter(q_real, q_pred_mean, color='blue', s=400, label='Data points')
        plt.plot([min(q_real), max(q_real)], [min(q_real), max(q_real)],
                 color='gray', linewidth=2, linestyle='dashed', label='Ideal')
        plt.xlabel('True ΔΔG (Query)', fontsize=46)
        plt.ylabel('Predicted ΔΔG (Query)', fontsize=46)
        plt.title(f'Query True vs Predicted ΔΔG - RF (anchor {ratio:.1f})', fontsize=46)
        plt.xlim(-5.5, 5); plt.ylim(-5.5, 5)
        plt.xticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
        plt.yticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
        plt.legend(loc='upper left', fontsize=42)
        plt.text(0.55, 0.12, f'Pearson: {pearson_corr:.2f}', transform=ax.transAxes, fontsize=44)
        plt.text(0.55, 0.05, f'Spearman: {spearman_corr:.2f}', transform=ax.transAxes, fontsize=44)
        plt.savefig(os.path.join(run_out_dir, 'RF_query_true_vs_pred.png'), dpi=900)
        plt.close()

    # 汇总 30 次
    if len(all_results_rows) > 0:
        final_df = pd.concat(all_results_rows, ignore_index=True)
        final_df.to_csv(os.path.join(subdir, 'RF_spearman_results.csv'), index=False)
        print(f"[ratio={ratio:.1f}] 汇总结果已保存：{os.path.join(subdir, 'RF_spearman_results.csv')}")
