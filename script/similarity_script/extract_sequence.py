import os
import pandas as pd

# ===== 路径 =====
csv_path = "/home/data1/BGM/fenziduiqi/check_embed_list.csv"
tsv_path = "/home/data1/BGM/fenziduiqi/MdrDB_mutation_output.tsv"
out_dir  = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence"
out_path = os.path.join(out_dir, "MMseqs2_check_embed_list.csv")

# ===== 读取数据 =====
df_csv = pd.read_csv(csv_path)
df_tsv = pd.read_csv(tsv_path, sep="\t")

# ===== 重命名列以便 merge =====
df_csv_renamed = df_csv.rename(columns={
    "PDB": "SAMPLE_PDB_ID",
    "uniprot_ids": "UNIPROT_ID",
    "Mutation": "MUTATION"
})

# ===== TSV 去重：同一键多条时只保留第一条 =====
df_tsv_unique = (
    df_tsv[["SAMPLE_PDB_ID", "UNIPROT_ID", "MUTATION", "sequence", "sequence_mutation"]]
    .drop_duplicates(subset=["SAMPLE_PDB_ID", "UNIPROT_ID", "MUTATION"], keep="first")
)

# ===== 合并（严格三列匹配）=====
df_merged = pd.merge(
    df_csv_renamed,
    df_tsv_unique,
    on=["SAMPLE_PDB_ID", "UNIPROT_ID", "MUTATION"],
    how="left",
    validate="m:1"  # 左表可多条，右表必须唯一
)

# ===== 恢复原始列名 =====
df_merged = df_merged.rename(columns={
    "SAMPLE_PDB_ID": "PDB",
    "UNIPROT_ID": "uniprot_ids",
    "MUTATION": "Mutation"
})

# ===== 保存结果 =====
os.makedirs(out_dir, exist_ok=True)
df_merged.to_csv(out_path, index=False)

print("✅ 合并完成")
print(f"原始 CSV 行数: {len(df_csv)}")
print(f"输出文件行数: {len(df_merged)}")
