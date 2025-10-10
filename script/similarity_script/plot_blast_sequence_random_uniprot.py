import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 路径 =====
csv_path_random  = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/blastp_full/test_vs_train_identity_full.csv"
csv_path_uniprot = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/blastp_uniprot_full/test_vs_train_identity_full.csv"

# ===== 读取数据 =====
df_random  = pd.read_csv(csv_path_random)
df_uniprot = pd.read_csv(csv_path_uniprot)

identities_random  = df_random["identity"].dropna()
identities_uniprot = df_uniprot["identity"].dropna()

# ===== 分箱区间 =====
bins = list(range(0, 101, 10))  # [0,10,...,100]
labels = [f"{bins[i+1]}%" for i in range(len(bins)-1)]  # 直接显示10%,20%,...

# ===== 统计每个区间的频率（概率）=====
counts_random, _  = np.histogram(identities_random, bins=bins)
counts_uniprot, _ = np.histogram(identities_uniprot, bins=bins)

prob_random  = counts_random  / counts_random.sum()
prob_uniprot = counts_uniprot / counts_uniprot.sum()

x = np.arange(len(labels))  # x轴位置
bar_width = 0.4

# ===== 绘制分组柱状图 =====
plt.figure(figsize=(8, 6.8))
plt.bar(x - bar_width/2, prob_random,  width=bar_width, label="Random split",  color="#97C8AF")
plt.bar(x + bar_width/2, prob_uniprot, width=bar_width, label="UniProt split", color="#96B6D8")

plt.xticks(x, labels, fontsize=24, rotation=45)
plt.ylabel("Probability", fontsize=28)
plt.xlabel("Sequence Identity", fontsize=28)
plt.legend(fontsize=24)

# ===== 设置 y 轴范围和刻度 =====
plt.ylim(0, 0.8)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=25)

# ===== 保存图片（固定路径）=====
out_img = "/home/data1/BGM/fenziduiqi/Blast_MdrDB_Sequence/plot_blast_random_uniprot.png"
plt.savefig(out_img, dpi=900, bbox_inches="tight")
plt.show()

print(f"✅ 概率直方图已保存: {out_img}")
