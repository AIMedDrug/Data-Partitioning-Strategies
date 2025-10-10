import pandas as pd

# ========= 路径 =========
fep_csv = "/home/bioinfor6/BGM/fenziduiqi/5zbq/5zbq_mutation_fep_with_sequences_add_pdb_smiles.csv"
embed_csv = "/home/bioinfor6/BGM/fenziduiqi/5zbq/embedding_5zbq_list.csv"
out_csv   = "/home/bioinfor6/BGM/fenziduiqi/5zbq/embedding_5zbq_list_with_label.csv"

# ========= 读取数据 =========
df_fep = pd.read_csv(fep_csv)
df_embed = pd.read_csv(embed_csv)

# 确认 fep_csv 里有 label 列
if "label" not in df_fep.columns:
    raise ValueError("fep_csv 文件中没有找到 'label' 列，请检查列名。")

# 为了匹配，把 fep 的 mutation 列重命名为 Mutation
df_fep = df_fep.rename(columns={"mutation": "Mutation"})

# ========= 合并 =========
df_merged = pd.merge(
    df_embed,
    df_fep[["PDB", "Mutation", "label"]],
    on=["PDB", "Mutation"],
    how="left"
)

# 保存新文件
df_merged.to_csv(out_csv, index=False)

print(f"完成！新文件已保存到: {out_csv}")
