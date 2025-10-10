import os
import pandas as pd
import random
import subprocess

# ===== 路径 =====
data_file = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/Blast_check_embed_list.csv"
work_dir = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/blastp_full"
os.makedirs(work_dir, exist_ok=True)

# ===== 读取数据 =====
df = pd.read_csv(data_file)

# 去掉缺失序列的行
df = df.dropna(subset=["sequence_mutation"])

# 给每条序列生成唯一 ID: PDB|Mutation
df["seq_id"] = df["PDB"].astype(str) + "|" + df["Mutation"].astype(str)

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

# ===== 保存为 FASTA =====
def save_fasta(df_subset, path):
    with open(path, "w") as f:
        for _, row in df_subset.iterrows():
            f.write(f">{row['seq_id']}\n{row['sequence_mutation']}\n")

train_fasta = os.path.join(work_dir, "train.fasta")
val_fasta   = os.path.join(work_dir, "val.fasta")
test_fasta  = os.path.join(work_dir, "test.fasta")
save_fasta(df_train, train_fasta)
save_fasta(df_val, val_fasta)
save_fasta(df_test, test_fasta)

# ===== 构建 BLAST 数据库（用 train 集）=====
train_db = os.path.join(work_dir, "train_db")
subprocess.run(f"makeblastdb -in {train_fasta} -dbtype prot -out {train_db}", shell=True, check=True)

# ===== 运行 blastp 全对全比对 =====
blast_out = os.path.join(work_dir, "test_vs_train.out")
cmd = (
    f"blastp -query {test_fasta} -db {train_db} "
    f"-outfmt '6 qseqid qlen sseqid slen pident' "
    f"-out {blast_out} -num_threads 8 -max_target_seqs 1000000"
)
subprocess.run(cmd, shell=True, check=True)

# ===== 读取结果并保存 CSV =====
results = []
with open(blast_out) as f:
    for line in f:
        qseqid, qlen, sseqid, slen, pident = line.strip().split("\t")
        results.append([qseqid, int(qlen), sseqid, int(slen), float(pident)])

df_res = pd.DataFrame(results, columns=["query_id", "query_len", "subject_id", "subject_len", "identity"])

out_csv = os.path.join(work_dir, "test_vs_train_identity_full.csv")
df_res.to_csv(out_csv, index=False)

print(f"✅ 已完成 blastp 全对全比对，结果保存在 {out_csv}")
