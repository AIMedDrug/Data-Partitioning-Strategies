import os
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ===== 路径设置 =====
embeddings_dir_1 = '/home/bioinfor6/BGM/fenziduiqi/5zbq/5zbq_embed'
embeddings_dir_2 = '/home/bioinfor6/BGM/fenziduiqi/5zbq/5zbq_mut_embed'
csv_file = '/home/bioinfor6/BGM/fenziduiqi/5zbq/5zbq_mutation_fep_with_sequences_add_pdb_smiles.csv'
output_dir = '/home/bioinfor6/BGM/fenziduiqi/5zbq'
os.makedirs(output_dir, exist_ok=True)

# ===== 加载 CSV 文件 =====
df = pd.read_csv(csv_file)

# 提取需要的数据列
pdb_ids = df['PDB'].values
mutation_all = df['mutation'].values
y = df['Exp'].values
smiles_all = df['SMILES'].values

print(f"加载的 CSV 文件行数: {len(df)}")
print("DataFrame 示例：")
print(df.head())

# ===== 定义计算 ECFP 指纹的函数 =====
def mol_from_smiles(smiles_str):
    mol = Chem.MolFromSmiles(str(smiles_str))
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fingerprint_array = fp.ToBitString()
        fingerprint_array = np.array([int(x) for x in fingerprint_array])
        return fingerprint_array
    else:
        print(f"无效的 SMILES: {smiles_str}")
        return np.zeros(1024, dtype=int)

# ===== 容器 =====
X1, X2 = [], []
filtered_y = []
filtered_pdb, filtered_muta, filtered_smiles = [], [], []
wild_lists, muta_lists, ecfp_fingerprints = [], [], []

# ===== 遍历数据并匹配 embedding =====
for i, (pdb_id, mutation) in enumerate(zip(pdb_ids, mutation_all)):
    wild_emb = f"{pdb_id}.pt"
    file_path_1 = os.path.join(embeddings_dir_1, wild_emb)

    muta_emb = f"{pdb_id}_{mutation}.pt"
    file_path_2 = os.path.join(embeddings_dir_2, muta_emb)

    if os.path.exists(file_path_1) and os.path.exists(file_path_2):
        filtered_pdb.append(pdb_id)
        filtered_muta.append(mutation)
        filtered_smiles.append(smiles_all[i])
        wild_lists.append(wild_emb)
        muta_lists.append(muta_emb)
        filtered_y.append(y[i])

        ecfp = mol_from_smiles(smiles_all[i])
        ecfp_fingerprints.append(np.array(ecfp))

        embedding_wild = torch.load(file_path_1)
        embedding_mut = torch.load(file_path_2)

        # 保证维度固定为 1280
        vec_wild = embedding_wild['mean_representations'][33].numpy().reshape(-1)[:1280]
        vec_mut  = embedding_mut['mean_representations'][33].numpy().reshape(-1)[:1280]

        # 如果不足 1280，补零
        if vec_wild.shape[0] < 1280:
            vec_wild = np.pad(vec_wild, (0, 1280 - vec_wild.shape[0]))
        if vec_mut.shape[0] < 1280:
            vec_mut = np.pad(vec_mut, (0, 1280 - vec_mut.shape[0]))

        X1.append(vec_wild)
        X2.append(vec_mut)
    else:
        print(f"缺少文件: {file_path_1} 或 {file_path_2}")

# ===== 保存结果 =====
filtered_df = pd.DataFrame({
    'PDB': filtered_pdb,
    'Mutation': filtered_muta,
    'smiles': filtered_smiles,
    'wild_emb': wild_lists,
    'muta_emb': muta_lists,
    'ddG': filtered_y,
    'X1': X1,
    'X2': X2,
    'ECFP': ecfp_fingerprints
})

# 转成字符串存储
filtered_df['ECFP'] = filtered_df['ECFP'].apply(lambda x: ','.join(map(str, x)))
filtered_df['X1']   = filtered_df['X1'].apply(lambda x: ','.join(map(str, x)))
filtered_df['X2']   = filtered_df['X2'].apply(lambda x: ','.join(map(str, x)))

output_file = os.path.join(output_dir, 'embedding_5zbq_list.csv')
filtered_df.to_csv(output_file, index=False)

print(f"保存完成: {output_file}")
print(filtered_df.head(1))
