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
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
root_output_dir = '/home/data1/BGM/fenziduiqi/All_X-_Data_pairing_UNIPROT_Q13315_RF_pred_result_0.1~0.9'
os.makedirs(root_output_dir, exist_ok=True)

# ====== 加载 CSV 文件 ======
df = pd.read_csv(csv_file)

# 如果没有列名则赋予默认列名，以保证后续引用正常
if 'UNIPROT' not in df.columns:
    df['UNIPROT'] = df.iloc[:, 6]
    df['ddG']     = df.iloc[:, 5]
    df['x1']      = df.iloc[:, 7]
    df['x2']      = df.iloc[:, 8]
    df['ECFP']    = df.iloc[:, 9]

# ====== 辅助函数：解析嵌入向量字符串 ======
def parse_emb(emb_str):
    """将逗号分隔的嵌入向量字符串转换为 numpy 数组"""
    return np.array([float(x) for x in emb_str.split(',')])

# ====== 函数：在同一 UNIPROT 内部做配对（Train 集内部配对） ======
def make_pairs(df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X    = []
    pairs_y    = []
    pairs_info = []
    grouped = df.groupby(uniprot_col)

    print("\n=== Train 集内部配对（打印每对 PDB 和 UNIPROT） ===")
    for uniprot, group in grouped:
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        for i, j in combinations(idxs, 2):
            d1 = group.loc[i]
            d2 = group.loc[j]
            x1 = np.concatenate([parse_emb(d1[x1_col]) - parse_emb(d1[x2_col]),
                                 parse_emb(d1[ecfp_col])])
            x2 = np.concatenate([parse_emb(d2[x1_col]) - parse_emb(d2[x2_col]),
                                 parse_emb(d2[ecfp_col])])
            X_pair = np.stack([x1, x2], axis=0)
            y_pair = d1[ddg_col] - d2[ddg_col]
            pairs_X.append(X_pair)
            pairs_y.append(y_pair)
            pairs_info.append((i, j, uniprot))
            print(f"  Train Pair: [idx1={i}, PDB={d1['PDB']}, UNIPROT={uniprot}]  <--> "
                  f"[idx2={j}, PDB={d2['PDB']}, UNIPROT={uniprot}]  | ΔΔG_diff={y_pair:.4f}")
    return pairs_X, pairs_y, pairs_info

# ====== 函数：在 Anchor 与 Query 之间做交叉配对 ======
def make_cross_pairs(df_anchor, df_query, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X    = []
    pairs_y    = []
    pairs_info = []
    if df_anchor[uniprot_col].iloc[0] != df_query[uniprot_col].iloc[0]:
        print(f"警告: Anchor UNIPROT 与 Query UNIPROT 不同: "
              f"{df_anchor[uniprot_col].iloc[0]} vs {df_query[uniprot_col].iloc[0]}")
    print("\n=== Query–Anchor 交叉配对（打印每对 PDB 和 UNIPROT） ===")
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
        print(f"  QA Pair: [Query idx={j}, PDB={d_query['PDB']}, UNIPROT={d_query[uniprot_col]}]  <--> "
              f"[Anchor idx={i}, PDB={d_anchor['PDB']}, UNIPROT={d_anchor[uniprot_col]}]  | ΔΔG_diff={y_pair:.4f}")
    return pairs_X, pairs_y, pairs_info

def df_all_text(df):
    return df.applymap(str)

def save_pairs_to_csv(pairs_X, pairs_y, pairs_info, output_path):
    X_flat = []
    for pair in pairs_X:
        x1 = ','.join(map(str, pair[0]))
        x2 = ','.join(map(str, pair[1]))
        X_flat.append([x1, x2])
    df_pairs = pd.DataFrame(X_flat, columns=['Sample1_features', 'Sample2_features'])
    df_pairs['ddG_diff'] = list(map(str, pairs_y))
    info_df = pd.DataFrame(pairs_info, columns=['idx1', 'idx2', 'UNIPROT']).applymap(str)
    df_out = pd.concat([df_pairs.applymap(str), info_df], axis=1)
    df_out.to_csv(output_path, index=False)

anchor_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ====== 主循环：遍历 anchor 比例和30个不同随机种子 ======
for ratio in anchor_ratios:
    for idx in range(30):
        random_number = random.randint(1, 9999999)
        print("随机种子：", random_number)

        # ====== 划分 Train/ Test（UNIPROT=Q13315 为 Test） ======
        test_df  = df[df['UNIPROT'] == 'Q13315'].copy()
        train_df = df[df['UNIPROT'] != 'Q13315'].copy()

        print(f"Total samples: {len(df)}")
        print(f"Test set (Q13315): {len(test_df)}")
        print(f"Train set (others): {len(train_df)}")

        # ====== 用本次的随机种子 ======
        random.seed(random_number)
        np.random.seed(random_number)

        test_indices  = test_df.index.tolist()
        anchor_size   = int(len(test_indices) * ratio)
        anchor_indices = random.sample(test_indices, anchor_size)
        query_indices  = list(set(test_indices) - set(anchor_indices))

        anchor_df = test_df.loc[anchor_indices].copy()
        query_df  = test_df.loc[query_indices].copy()

        # ====== 合并训练集（Train + Anchor） ======
        combined_train_df = pd.concat([train_df, anchor_df], axis=0)

        print(f"Anchor set size: {len(anchor_df)}")
        print(f"Query set size: {len(query_df)}")
        print(f"Train set (train+anchor): {len(combined_train_df)}")

        print("\n【Anchor 详情】")
        for idx_, row in anchor_df.iterrows():
            print(f"  idx={idx_}\tPDB={row['PDB']}\tUNIPROT={row['UNIPROT']}\tddG={row['ddG']:.4f}")
        print("\n【Query 详情】")
        for idx_, row in query_df.iterrows():
            print(f"  idx={idx_}\tPDB={row['PDB']}\tUNIPROT={row['UNIPROT']}\tddG={row['ddG']:.4f}")

        # ====== 指定各列名 ======
        x1_col       = 'x1'
        x2_col       = 'x2'
        ecfp_col     = 'ECFP'
        ddg_col      = 'ddG'
        uniprot_col  = 'UNIPROT'

        # ====== 生成配对数据 ======
        print("\n开始配对...")
        tX, ty, tinfo   = make_pairs(combined_train_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)
        aqX, aqy, aqinfo = make_cross_pairs(anchor_df, query_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)

        print(f"\n配对完成：")
        print(f"  Train pairs: {len(tX)}")
        print(f"  Query-Anchor pairs: {len(aqX)} （预期约为 {len(anchor_df)} × {len(query_df)} = {len(anchor_df) * len(query_df)}）")

        # ====== 子文件夹输出路径 ======
        ratio_str = str(ratio).replace('.', '')
        output_dir = os.path.join(root_output_dir, f'All_X-_Data_pairing_UNIPROT_Q13315_RF_pred_result_{ratio}_seed{random_number}')
        os.makedirs(output_dir, exist_ok=True)

        # ====== 保存 NPY 和 INFO 文件 ======
        #np.save(os.path.join(output_dir, 'train_pairs_X.npy'), np.array(tX))
        #np.save(os.path.join(output_dir, 'train_pairs_y.npy'), np.array(ty))
        #np.save(os.path.join(output_dir, 'query_anchor_pairs_X.npy'), np.array(aqX))
        #np.save(os.path.join(output_dir, 'query_anchor_pairs_y.npy'), np.array(aqy))

        df_all_text(pd.DataFrame(tinfo, columns=['idx1','idx2','UNIPROT'])).to_csv(
            os.path.join(output_dir, 'train_pairs_info.csv'), index=False)
        df_all_text(pd.DataFrame(aqinfo, columns=['idx_query','idx_anchor','UNIPROT'])).to_csv(
            os.path.join(output_dir, 'query_anchor_pairs_info.csv'), index=False)

        #save_pairs_to_csv(tX, ty, tinfo,
                          #os.path.join(output_dir, 'train_pairs_data.csv'))
        #save_pairs_to_csv(aqX, aqy, aqinfo,
                          #os.path.join(output_dir, 'query_anchor_pairs_data.csv'))

        print("所有配对数据已保存到 CSV 和 NPY 文件。")

        # ====== RF 模型训练与预测 ======
        print("\n开始 RF 模型训练和预测...")

        def prepare_data_for_rf(pairs_X):
            n_pairs = len(pairs_X)
            if n_pairs == 0:
                return np.array([])
            feature_dim = pairs_X[0].shape[1]
            X_flat = np.zeros((n_pairs, 2 * feature_dim))
            for idx, pair in enumerate(pairs_X):
                X_flat[idx] = np.concatenate([pair[0], pair[1]])
            return X_flat

        X_train = prepare_data_for_rf(tX)
        y_train = np.array(ty)
        X_test  = prepare_data_for_rf(aqX)
        y_test  = np.array(aqy)

        print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")

        # 随机森林超参数
        n_estimators       = 100
        max_depth          = 20
        min_samples_split  = 2

        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )

        print("训练 RF 模型...")
        rf_model.fit(X_train, y_train)
        print("模型训练完成。")

        print("在测试集上进行预测...")
        y_pred = rf_model.predict(X_test)

        # 计算各项评估指标
        mae, mse = mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred)
        rmse     = np.sqrt(mse)
        pearson_corr, _  = pearsonr(y_test, y_pred)
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
        # 已去除 True Mean 和 Pred Mean 标注

        figname    = f'RF_query_true_vs_pred_estimators{n_estimators}_depth{max_depth}_split{min_samples_split}_query_anchor.png'
        output_path = os.path.join(output_dir, figname)
        plt.savefig(output_path, dpi=900)
        plt.close()

        print(f"Query 真实-预测值散点图已保存到: {output_path}")
        print("模型预测和评估完成。")
