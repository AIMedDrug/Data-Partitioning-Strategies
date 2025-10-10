import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd  # 可选：替换为 import cudf as pd 以进一步加速数据加载
import numpy as np
from itertools import combinations, product
# from sklearn.ensemble import RandomForestRegressor  # 注释掉
from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor  # 新增：cuML 版本
from sklearn.metrics import mean_squared_error, mean_absolute_error  # metrics 可保持 sklearn
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ====== 路径设置 ======
train_csv = '/home/bioinfor6/BGM/fenziduiqi/check_embed_list.csv'
test_csv_full  = '/home/bioinfor6/BGM/fenziduiqi/5zbq/embedding_5zbq_list_with_label.csv'
root_output_dir = '/home/bioinfor6/BGM/fenziduiqi/5zbq/RF_pred_result_unpaired'
os.makedirs(root_output_dir, exist_ok=True)

# ====== 加载数据 ======
train_df_base = pd.read_csv(train_csv)
test_df_full = pd.read_csv(test_csv_full)

# 数据划分：将 test_csv_full 中 label 为 'ref' 的行添加到 train，其余作为 test
ref_mask = test_df_full['label'] == 'ref'
ref_rows = test_df_full[ref_mask].copy()
test_rows = test_df_full[~ref_mask].copy()

train_df = pd.concat([train_df_base, ref_rows], ignore_index=True)
test_df = test_rows

print(f"Ref rows added to train: {len(ref_rows)}")
print(f"Final train size: {len(train_df)}")
print(f"Final test size: {len(test_df)}")

# ====== 兼容列名 ======
def normalize_columns(df):
    lower_map = {col.lower(): col for col in df.columns}
    def get_col(*possible_names):
        for name in possible_names:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    col_uniprot = get_col('UNIPROT', 'Uniprot', 'uniprot_id', 'uniprot_ids')
    col_ddg     = get_col('ddG', 'ddg', 'DDG', 'Label')
    col_x1      = get_col('x1', 'X1', 'esm1', 'embedding1')
    col_x2      = get_col('x2', 'X2', 'esm2', 'embedding2')
    col_ecfp    = get_col('ecfp', 'ECFP', 'smiles_fp', 'fingerprint')

    # 自动补充统一列名
    if col_uniprot and 'UNIPROT' not in df.columns:
        df['UNIPROT'] = df[col_uniprot]
    if col_ddg and 'ddG' not in df.columns:
        df['ddG'] = df[col_ddg]
    if col_x1 and 'x1' not in df.columns:
        df['x1'] = df[col_x1]
    if col_x2 and 'x2' not in df.columns:
        df['x2'] = df[col_x2]
    if col_ecfp and 'ECFP' not in df.columns:
        df['ECFP'] = df[col_ecfp]

    return df

# 应用到两个表
train_df = normalize_columns(train_df)
test_df  = normalize_columns(test_df)


# ====== 工具函数 ======
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

def check_and_filter(df, name):
    """检查并删除不符合维度的行，并输出详细原因"""
    print(f"\n==== 检查 {name} 维度 ====")
    for col, expected_dim in [("x1", 1280), ("x2", 1280), ("ECFP", 1024)]:
        lengths = df[col].dropna().apply(lambda x: len(x.split(',')))
        bad_idx = lengths[lengths != expected_dim].index

        if len(bad_idx) > 0:
            print(f"⚠️ {col} 中有 {len(bad_idx)} 行维度不等于 {expected_dim}，打印前几条详细信息：")

            # 打印前几条错误信息（可调整数量）
            for i in bad_idx[:10]:
                raw_str = str(df.loc[i, col])
                true_len = len(raw_str.split(','))
                pdb = df.loc[i, 'PDB'] if 'PDB' in df.columns else 'N/A'
                mut = df.loc[i, 'Mutation'] if 'Mutation' in df.columns else 'N/A'
                lbl = df.loc[i, 'label'] if 'label' in df.columns else 'N/A'
                print(f"  行号 {i}: 实际维度={true_len}, PDB={pdb}, Mutation={mut}, Label={lbl}")
                if true_len == 1 and (raw_str.strip() == '' or raw_str.strip() == '0'):
                    print(f"    ⚠️ 内容为空或损坏: '{raw_str[:30]}'")
                else:
                    print(f"    内容片段: '{raw_str[:60]}...'")

            # 保存所有不合格行以便分析
            bad_save_path = f"{name}_{col}_bad_rows.csv"
            df.loc[bad_idx].to_csv(os.path.join(root_output_dir, bad_save_path), index=False)
            print(f"⏬ 已保存不合格行至: {bad_save_path}")

            # 删除这些错误行
            df = df.drop(bad_idx)
        else:
            print(f"✓ {col} 全部符合 {expected_dim} 维度。")

    return df


# 清洗数据
train_df = check_and_filter(train_df, "Train")
test_df  = check_and_filter(test_df, "Test")

def prepare_single_data(df, x1_col, x2_col, ecfp_col, ddg_col):
    """准备单个样本数据（非配对）"""
    samples_X, samples_y, samples_info = [], [], []
    for idx in df.index:
        try:
            v1 = parse_emb(df.loc[idx, x1_col])
            v2 = parse_emb(df.loc[idx, x2_col])
            ecfp = parse_emb(df.loc[idx, ecfp_col])
        except:
            continue
        # 维度检查
        if v1.shape[0] != 1280 or v2.shape[0] != 1280:
            continue
        if ecfp.shape[0] != 1024:
            continue
        # 拼接向量：X1 - X2 后与 ECFP 拼接
        diff = v1 - v2
        X_single = np.concatenate([diff, ecfp])
        y_single = df.loc[idx, ddg_col]
        samples_X.append(X_single)
        samples_y.append(y_single)
        samples_info.append(idx)
    return samples_X, samples_y, samples_info

def prepare_data_for_rf(samples_X):
    return np.array(samples_X)

# ====== 生成单个样本数据 ======
print("开始生成训练单个样本...")
tX, ty, tinfo = prepare_single_data(train_df, "x1", "x2", "ECFP", "ddG")
print(f"训练样本数量: {len(tX)}")

print("开始生成测试单个样本...")
aqX, aqy, aqinfo = prepare_single_data(test_df, "x1", "x2", "ECFP", "ddG")
print(f"测试样本数量: {len(aqX)}")

# 保存样本信息
sample_info_dir = os.path.join(root_output_dir, "sample_info")
os.makedirs(sample_info_dir, exist_ok=True)
pd.DataFrame(tinfo, columns=['idx']).to_csv(
    os.path.join(sample_info_dir, "train_samples_info.csv"), index=False)
pd.DataFrame(aqinfo, columns=['idx']).to_csv(
    os.path.join(sample_info_dir, "test_samples_info.csv"), index=False)

# ====== 准备数据 ======
X_train = prepare_data_for_rf(tX)
y_train = np.array(ty)
X_test  = prepare_data_for_rf(aqX)
y_test  = np.array(aqy)

# 测试集为空保护
if len(aqX) == 0:
    print("  ⚠️ 测试集样本为空，跳过预测。")
else:
    print(f"  训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # RF 超参数（固定）
    n_est = 100
    max_d = 20
    min_s = 10

    print(f"\n===== 参数组合: n_estimators={n_est}, max_depth={max_d}, min_samples_split={min_s} =====")
    
    combo_dir = os.path.join(root_output_dir, f"n{n_est}_d{max_d}_s{min_s}")
    os.makedirs(combo_dir, exist_ok=True)
    
    pearson_rmse_spearman_summary = []
    detailed_results = []

    for repeat_idx in range(30):
        print(f"  第 {repeat_idx + 1} 次模型训练")
        repeat_dir = os.path.join(combo_dir, f'result_{repeat_idx+1}')
        os.makedirs(repeat_dir, exist_ok=True)

        # RF模型
        rf_model = CuMLRandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_s,
            random_state=repeat_idx
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        # 指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)

        print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}")

        # 保存预测结果
        results_df = pd.DataFrame({
            'True_ddG': y_test,
            'Predicted_ddG': y_pred
        })
        results_df.to_csv(os.path.join(repeat_dir, "RF_prediction_results.csv"), index=False)

        # 保存图
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
        plt.scatter(y_test, y_pred, color='blue', label='Data points', s=400)
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
        pearson_q, _  = pearsonr(y_test, y_pred)
        spearman_q, _ = spearmanr(y_test, y_pred)
        plt.text(
            0.55, 0.12, f'Pearson: {pearson_q:.2f}',
            transform=plt.gca().transAxes, fontsize=44
        )
        plt.text(
            0.55, 0.05, f'Spearman: {spearman_q:.2f}',
            transform=plt.gca().transAxes, fontsize=44
        )
        plt.savefig(os.path.join(repeat_dir, "RF_test_true_vs_pred.png"), dpi=600)
        plt.close()

        # 汇总
        pearson_rmse_spearman_summary.append({
            "repeat_idx": repeat_idx+1,
            "MAE": mae,
            "RMSE": rmse,
            "Pearson": pearson_corr,
            "Spearman": spearman_corr
        })

        # 收集详细结果（针对测试样本）
        for k in range(len(aqinfo)):
            idx = aqinfo[k]
            row = test_df.loc[idx]
            pdb = row.get('PDB', 'N/A')
            mut = row.get('Mutation', 'N/A')
            uniprot = row.get('UNIPROT', row.get('label', 'N/A'))  # 优先UNIPROT，否则用label
            ddg = row['ddG']
            true_ddg = y_test[k]
            pred_ddg = y_pred[k]
            detailed_results.append({
                'PDB': pdb,
                'Mutation': mut,
                'UNIPROT': uniprot,
                'ddG': ddg,
                'True_ddG': true_ddg,
                'Pred_ddG': pred_ddg,
                'Random_Number': repeat_idx + 1,
                'Pearson_Corr': pearson_corr,
                'Spearman_Corr': spearman_corr
            })

    # ====== 保存当前参数组合的30次结果汇总 ======
    summary_df = pd.DataFrame(pearson_rmse_spearman_summary)
    summary_df.to_csv(os.path.join(combo_dir, "test_samples_pearson_rmse_spearman_summary.csv"), index=False)
    print(f"  组合 n{n_est}_d{max_d}_s{min_s}: 所有30次的MAE、Pearson、RMSE和Spearman已保存。")

    # ====== 保存当前参数组合的详细输出文件 ======
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(os.path.join(combo_dir, "detailed_test_results.csv"), index=False)
    print(f"  组合 n{n_est}_d{max_d}_s{min_s}: 详细测试结果（包括PDB, Mutation, UNIPROT, ddG, Pred_ddG等）已保存至 detailed_test_results.csv。")

print("运行完成。")