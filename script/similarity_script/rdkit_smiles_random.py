import os
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ===== 路径 =====
data_file = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/Blast_check_embed_list.csv"
work_dir = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/smiles_similarity"
os.makedirs(work_dir, exist_ok=True)

# ===== 读取数据 =====
df = pd.read_csv(data_file)

# 去掉缺失 SMILES 的行
df = df.dropna(subset=["smiles"])

# 给每条 SMILES 生成唯一 ID: PDB|Mutation
df["mol_id"] = df["PDB"].astype(str) + "|" + df["Mutation"].astype(str)

# ===== 8:1:1 划分 =====
all_idx = list(df.index)
random.shuffle(all_idx)
n = len(all_idx)
train_idx = all_idx[: int(0.8 * n)]
val_idx   = all_idx[int(0.8 * n): int(0.9 * n)]
test_idx  = all_idx[int(0.9 * n):]

df_train = df.loc[train_idx]
df_val   = df.loc[val_idx]
df_test  = df.loc[test_idx]

# ===== 分子对象和指纹计算 =====
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

train_fps = {row["mol_id"]: smiles_to_fp(row["smiles"]) for _, row in df_train.iterrows()}
test_fps  = {row["mol_id"]: smiles_to_fp(row["smiles"]) for _, row in df_test.iterrows()}

# ===== 相似性计算 (Tanimoto) =====
results = []
for qid, qfp in test_fps.items():
    if qfp is None:
        continue
    for sid, sfp in train_fps.items():
        if sfp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(qfp, sfp)
        results.append([qid, sid, sim * 100])  # 转为百分比

# ===== 保存结果 =====
df_res = pd.DataFrame(results, columns=["query_id", "subject_id", "similarity(%)"])
out_csv = os.path.join(work_dir, "test_vs_train_smiles_similarity.csv")
df_res.to_csv(out_csv, index=False)

print(f"✅ 已完成 SMILES 相似性计算，结果保存在 {out_csv}")
