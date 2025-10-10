import os
import pandas as pd
import numpy as np
import random
from itertools import combinations, product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ====== 路径设置 ======
csv_file = '/home/bioinfor6/BGM/fenziduiqi/check_embed_list.csv'
root_output_dir = '/home/bioinfor6/BGM/fenziduiqi/All_X-_Data_pairing_UNIPROT_Q13315_Transformer_pred_result_0.1~0.9'
os.makedirs(root_output_dir, exist_ok=True)

# ====== 加载 CSV 文件 ======
df = pd.read_csv(csv_file)
if 'UNIPROT' not in df.columns:
    df['UNIPROT'] = df.iloc[:, 6]
    df['ddG']     = df.iloc[:, 5]
    df['x1']      = df.iloc[:, 7]
    df['x2']      = df.iloc[:, 8]
    df['ECFP']    = df.iloc[:, 9]

# ====== 辅助函数 ======
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

def make_pairs(df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X, pairs_y, pairs_info = [], [], []
    grouped = df.groupby(uniprot_col)
    print("\n=== Train 集内部配对（打印每对 PDB 和 UNIPROT） ===")
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
            print(f"  Train Pair: [idx1={i}, PDB={d1['PDB']}, UNIPROT={uniprot}]  <--> [idx2={j}, PDB={d2['PDB']}, UNIPROT={uniprot}]  | ΔΔG_diff={y_pair:.4f}")
    return pairs_X, pairs_y, pairs_info

def make_cross_pairs(df_anchor, df_query, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col):
    pairs_X, pairs_y, pairs_info = [], [], []
    if df_anchor[uniprot_col].iloc[0] != df_query[uniprot_col].iloc[0]:
        print(f"警告: Anchor UNIPROT 与 Query UNIPROT 不同: {df_anchor[uniprot_col].iloc[0]} vs {df_query[uniprot_col].iloc[0]}")
    print("\n=== Query–Anchor 交叉配对（打印每对 PDB 和 UNIPROT） ===")
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
        print(f"  QA Pair: [Query idx={j}, PDB={d_query['PDB']}, UNIPROT={d_query[uniprot_col]}]  <--> [Anchor idx={i}, PDB={d_anchor['PDB']}, UNIPROT={d_anchor[uniprot_col]}]  | ΔΔG_diff={y_pair:.4f}")
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

# ====== Transformer 数据集 ======
class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 定义 Transformer 模型 ======
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=1, dropout=0.3, nhead=4):
        super(TransformerRegressor, self).__init__()
        # input_dim即特征数，也是d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len=2, input_dim]
        x = self.transformer_encoder(x)        # [batch, seq_len, input_dim]
        x = x[:, -1, :]                        # 取最后一个时间步的输出 [batch, input_dim]
        x = torch.relu(self.fc(x))             # [batch, hidden_dim]
        x = self.dropout(x)
        x = self.fc_final(x)                   # [batch, 1]
        return x.squeeze(-1)                   # [batch]

# Transformer 固定参数
hidden_dim = 1024
dropout = 0.3
num_layers = 1
nhead = 4

anchor_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for ratio in anchor_ratios:
    for idx in range(30):
        random_number = random.randint(1, 9999999)
        print("随机种子：", random_number)

        # 划分 Train/ Test
        test_df  = df[df['UNIPROT'] == 'Q13315'].copy()
        train_df = df[df['UNIPROT'] != 'Q13315'].copy()
        print(f"Total samples: {len(df)}")
        print(f"Test set (Q13315): {len(test_df)}")
        print(f"Train set (others): {len(train_df)}")

        # 用本次随机种子
        random.seed(random_number)
        np.random.seed(random_number)
        test_indices  = test_df.index.tolist()
        anchor_size   = int(len(test_indices) * ratio)
        anchor_indices = random.sample(test_indices, anchor_size)
        query_indices  = list(set(test_indices) - set(anchor_indices))
        anchor_df = test_df.loc[anchor_indices].copy()
        query_df  = test_df.loc[query_indices].copy()
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

        x1_col       = 'x1'
        x2_col       = 'x2'
        ecfp_col     = 'ECFP'
        ddg_col      = 'ddG'
        uniprot_col  = 'UNIPROT'

        print("\n开始配对...")
        tX, ty, tinfo   = make_pairs(combined_train_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)
        aqX, aqy, aqinfo = make_cross_pairs(anchor_df, query_df, x1_col, x2_col, ecfp_col, ddg_col, uniprot_col)
        print(f"\n配对完成：")
        print(f"  Train pairs: {len(tX)}")
        print(f"  Query-Anchor pairs: {len(aqX)} （预期约为 {len(anchor_df)} × {len(query_df)} = {len(anchor_df) * len(query_df)}）")

        ratio_str = str(ratio).replace('.', '')
        output_dir = os.path.join(root_output_dir, f'All_X-_Data_pairing_UNIPROT_Q13315_Transformer_pred_result_{ratio}_seed{random_number}')
        os.makedirs(output_dir, exist_ok=True)

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

        # ====== Transformer 模型训练与预测 ======
        print("\n开始 Transformer 模型训练和预测...")

        def prepare_data_for_tf(pairs_X, pairs_y):
            X = np.stack(pairs_X, axis=0)
            y = np.array(pairs_y)
            return X, y

        X_train, y_train = prepare_data_for_tf(tX, ty)
        X_test, y_test = prepare_data_for_tf(aqX, aqy)
        train_dataset = PairDataset(X_train, y_train)
        test_dataset  = PairDataset(X_test, y_test)
        train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader   = DataLoader(test_dataset, batch_size=128, shuffle=False)

        input_dim = X_train.shape[2]
        model = TransformerRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        num_epochs = 20

        model.train()
        for epoch in range(num_epochs):
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses):.4f}")

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                y_pred = model(xb).cpu().numpy()
                preds.append(y_pred)
        y_pred = np.concatenate(preds, axis=0)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr, _  = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)

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

        results_df = pd.DataFrame({
            'True_ddG_diff'      : y_test,
            'Predicted_ddG_diff' : y_pred
        }).applymap(str)
        results_df.to_csv(os.path.join(output_dir, 'Transformer_prediction_results.csv'), index=False)

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
        avg_df.to_csv(os.path.join(output_dir, 'Transformer_prediction_averages.csv'), index=False)

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
        plt.title('Query True vs Predicted ΔΔG - TF', fontsize=46)
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
        # True Mean/Pred Mean注释已移除
        figname    = f'Transformer_query_true_vs_pred_hid{hidden_dim}_drop{dropout}_layers{num_layers}_query_anchor.png'
        output_path = os.path.join(output_dir, figname)
        plt.savefig(output_path, dpi=900)
        plt.close()

        print(f"Query 真实-预测值散点图已保存到: {output_path}")
        print("模型预测和评估完成。")
