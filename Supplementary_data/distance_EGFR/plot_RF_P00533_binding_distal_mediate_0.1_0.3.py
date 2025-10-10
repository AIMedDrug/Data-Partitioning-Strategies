import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# =============== 路径 ===============
base_dir = "/home/bioinfor6/BGM/fenziduiqi"

# random 结果
all_file     = os.path.join(base_dir, "All_X-_RF_P00533_random_result",          "RF_spearman_results.csv")
binding_file = os.path.join(base_dir, "All_X-_RF_P00533_random_binding_result",  "RF_spearman_results.csv")
distal_file  = os.path.join(base_dir, "All_X-_RF_P00533_random_distal_result",   "RF_spearman_results.csv")
mediate_file = os.path.join(base_dir, "All_X-_RF_P00533_random_mediate_result",  "RF_spearman_results.csv")

# ratio_10 / ratio_30 汇总
ratio10_csv = os.path.join(base_dir, "All_X-_Data_pairing_P00533_RF_result_0.1_0.3_Analyze", "ratio_10", "RF_spearman_summary_by_category.csv")
ratio30_csv = os.path.join(base_dir, "All_X-_Data_pairing_P00533_RF_result_0.1_0.3_Analyze", "ratio_30", "RF_spearman_summary_by_category.csv")

# =============== 读取数据 ===============
all_df     = pd.read_csv(all_file)
binding_df = pd.read_csv(binding_file)
distal_df  = pd.read_csv(distal_file)
mediate_df = pd.read_csv(mediate_file)

ratio10_df = pd.read_csv(ratio10_csv)
ratio30_df = pd.read_csv(ratio30_csv)

# =============== 计算均值 ===============
# Mix
mix_random = all_df["Spearman_Corr"].mean()
mix_r10    = ratio10_df["Spearman_All"].mean()
mix_r30    = ratio30_df["Spearman_All"].mean()

# pocket
pocket_random = binding_df["Spearman_Corr"].mean()
pocket_r10    = ratio10_df["Spearman_binding pocket"].mean()
pocket_r30    = ratio30_df["Spearman_binding pocket"].mean()

# distal
distal_random = distal_df["Spearman_Corr"].mean()
distal_r10    = ratio10_df["Spearman_distal allosteric"].mean()
distal_r30    = ratio30_df["Spearman_distal allosteric"].mean()

# IMD
imd_random = mediate_df["Spearman_Corr"].mean()
imd_r10    = ratio10_df["Spearman_intermediate"].mean()
imd_r30    = ratio30_df["Spearman_intermediate"].mean()

# =============== 画图（分组柱状图） ===============
labels = ["Mix", "pocket", "distal", "IMD"]
x = np.arange(len(labels))

# 柱子宽度和间隔
bar_width = 0.23
gap = 0.05
offsets = [-bar_width-gap, 0.0, bar_width+gap]

plt.figure(figsize=(6, 4.8))

# 三个颜色
color_random = "#78aac8"
color_r10    = "#7dba7f"
color_r30    = "#ac491a"

# Mix
plt.bar(x[0] + offsets[0], mix_random, width=bar_width, color=color_random)
plt.bar(x[0] + offsets[1], mix_r10,    width=bar_width, color=color_r10)
plt.bar(x[0] + offsets[2], mix_r30,    width=bar_width, color=color_r30)

# pocket
plt.bar(x[1] + offsets[0], pocket_random, width=bar_width, color=color_random)
plt.bar(x[1] + offsets[1], pocket_r10,    width=bar_width, color=color_r10)
plt.bar(x[1] + offsets[2], pocket_r30,    width=bar_width, color=color_r30)

# distal
plt.bar(x[2] + offsets[0], distal_random, width=bar_width, color=color_random)
plt.bar(x[2] + offsets[1], distal_r10,    width=bar_width, color=color_r10)
plt.bar(x[2] + offsets[2], distal_r30,    width=bar_width, color=color_r30)

# IMD
plt.bar(x[3] + offsets[0], imd_random, width=bar_width, color=color_random)
plt.bar(x[3] + offsets[1], imd_r10,    width=bar_width, color=color_r10)
plt.bar(x[3] + offsets[2], imd_r30,    width=bar_width, color=color_r30)

plt.xticks(x, labels, rotation=45, fontsize=20)
plt.ylabel("Spearman", fontsize=22)
plt.title("EGFR", fontsize=22)

plt.ylim(0, 0.7)
plt.yticks([0, 0.2, 0.4, 0.6], fontsize=20)

# ===== 图例竖排左上角，颜色与柱子保持一致 =====
handles = [
    plt.Rectangle((0,0),1,1,color=color_random),
    plt.Rectangle((0,0),1,1,color=color_r10),
    plt.Rectangle((0,0),1,1,color=color_r30)
]
labels_legend = ["random_no_pair", "ratio=10%_pair", "ratio=30%_pair"]

plt.legend(
    handles, labels_legend,
    fontsize=15,
    frameon=False,
    ncol=1,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98)
)

plt.tight_layout()

output_file = os.path.join(base_dir, "plot_RF_P00533_binding_distal_mediate_0.1_0.3.png")
plt.savefig(output_file, dpi=900)
plt.close()
print(f"分组柱状图已保存到: {output_file}")
