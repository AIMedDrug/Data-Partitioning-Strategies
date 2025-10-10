import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# =============== 路径 ===============
base_dir = "/home/bioinfor6/BGM/fenziduiqi"

main_csv   = os.path.join(base_dir, "All_X-_Data_pair_P00520_p00519_RF_pred_Analyze_result",
                          "query_pred_ddG_means_spearman_summary.csv")
ratio10_csv = os.path.join(base_dir, "All_X-_Data_pair_P00520_p00519_RF_pred_Analyze_0.1_0.3_result",
                           "ratio0.1", "RF_spearman_summary_by_category.csv")
ratio30_csv = os.path.join(base_dir, "All_X-_Data_pair_P00520_p00519_RF_pred_Analyze_0.1_0.3_result",
                           "ratio0.3", "RF_spearman_summary_by_category.csv")

# =============== 读取数据 ===============
df_main   = pd.read_csv(main_csv)
df_r10    = pd.read_csv(ratio10_csv)
df_r30    = pd.read_csv(ratio30_csv)

# =============== 计算均值 ===============
# Mix
mix_main = df_main["All"].mean()
mix_r10  = df_r10["Spearman_All"].mean()
mix_r30  = df_r30["Spearman_All"].mean()

# pocket
pocket_main = df_main["binding pocket"].mean()
pocket_r10  = df_r10["Spearman_binding pocket"].mean()
pocket_r30  = df_r30["Spearman_binding pocket"].mean()

# distal
distal_main = df_main["distal allosteric"].mean()
distal_r10  = df_r10["Spearman_distal allosteric"].mean()
distal_r30  = df_r30["Spearman_distal allosteric"].mean()

# IMD
imd_main = df_main["intermediate"].mean()
imd_r10  = df_r10["Spearman_intermediate"].mean()
imd_r30  = df_r30["Spearman_intermediate"].mean()

# =============== 画图（分组柱状图） ===============
labels = ["Mix", "pocket", "distal", "IMD"]
x = np.arange(len(labels))

bar_width = 0.23
gap = 0.05
offsets = [-bar_width - gap, 0.0, bar_width + gap]

plt.figure(figsize=(6, 4.8))

color_main = "#78aac8"   # random_pair
color_r10  = "#7dba7f"   # ratio=10%_pair
color_r30  = "#ac491a"   # ratio=30%_pair

# Mix
plt.bar(x[0] + offsets[0], mix_main, width=bar_width, color=color_main)
plt.bar(x[0] + offsets[1], mix_r10,  width=bar_width, color=color_r10)
plt.bar(x[0] + offsets[2], mix_r30,  width=bar_width, color=color_r30)

# pocket
plt.bar(x[1] + offsets[0], pocket_main, width=bar_width, color=color_main)
plt.bar(x[1] + offsets[1], pocket_r10,  width=bar_width, color=color_r10)
plt.bar(x[1] + offsets[2], pocket_r30,  width=bar_width, color=color_r30)

# distal
plt.bar(x[2] + offsets[0], distal_main, width=bar_width, color=color_main)
plt.bar(x[2] + offsets[1], distal_r10,  width=bar_width, color=color_r10)
plt.bar(x[2] + offsets[2], distal_r30,  width=bar_width, color=color_r30)

# IMD
plt.bar(x[3] + offsets[0], imd_main, width=bar_width, color=color_main)
plt.bar(x[3] + offsets[1], imd_r10,   width=bar_width, color=color_r10)
plt.bar(x[3] + offsets[2], imd_r30,   width=bar_width, color=color_r30)

plt.xticks(x, labels, rotation=45, fontsize=20)
plt.ylabel("Spearman", fontsize=22)
plt.title("ABL Kinase", fontsize=22)

plt.ylim(-0.2, 0.2)
plt.yticks([-0.2, -0.1, 0, 0.1, 0.2], fontsize=20)

# ===== 在 Y=0 画黑色实线 =====
ax = plt.gca()
axis_linewidth = ax.spines['left'].get_linewidth()
plt.axhline(y=0, color="black", linewidth=axis_linewidth)

# ===== 图例竖排右上角 =====
handles = [
    plt.Rectangle((0,0),1,1,color=color_main),
    plt.Rectangle((0,0),1,1,color=color_r10),
    plt.Rectangle((0,0),1,1,color=color_r30)
]
labels_legend = ["random_pair", "ratio=10%_pair", "ratio=30%_pair"]

plt.legend(handles, labels_legend, fontsize=14, frameon=False,
           ncol=1, loc="upper right", bbox_to_anchor=(1, 1))

plt.tight_layout()

output_file = os.path.join(base_dir, "plot_RF_P00520_P00519_mix_bind_distal_mediate_0.1_0.3.png")
plt.savefig(output_file, dpi=900)
plt.close()
print(f"分组柱状图已保存到: {output_file}")
