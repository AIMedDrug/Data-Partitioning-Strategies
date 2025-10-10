import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ======= 输入文件路径 =======
esm2_csv = Path("/home/bioinfor6/BGM/fenziduiqi/ESMC/ESM_RF_best_results.csv")
esmc_csv = Path("/home/bioinfor6/BGM/fenziduiqi/ESMC/All_X-_4179_RF_best_ESMC_result/RF_best_results.csv")
protT5_csv = Path("/home/bioinfor6/BGM/fenziduiqi/ProtT5/All_X-_4179_RF_best_ProtT5_result/RF_best_results.csv")

# ======= 读取并取 MAE 的均值和标准差 =======
def read_mae_stats(csv_path: Path, col="Spearman"):
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise KeyError(f"{csv_path} 中找不到列：{col}")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        raise ValueError(f"{csv_path} 的 {col} 列为空或无法解析为数值。")
    return vals.mean(), vals.std()

esm2_mean, esm2_std = read_mae_stats(esm2_csv)
esmc_mean, esmc_std = read_mae_stats(esmc_csv)
protT5_mean, protT5_std = read_mae_stats(protT5_csv)

labels = ["ESM-2", "ESMC", "ProtT5"]
means = [esm2_mean, esmc_mean, protT5_mean]
stds = [esm2_std, esmc_std, protT5_std]

# ======= 作图 =======
plt.figure(figsize=(6, 4.8))

x = np.arange(len(labels)) * 0.4  
bar_width = 0.2

# 画柱子 + SD 误差棒
bars = plt.bar(
    x, means,
    yerr=stds, capsize=8,
    color=["#4B9CD3", "#3EB489", "#F7B7C5"],
    width=bar_width,
    error_kw=dict(ecolor="black", lw=6)  # 误差棒样式（稍微加粗）
)

# 固定 Y 轴范围和刻度
plt.ylim(0, 0.7)
plt.yticks([0.1, 0.3, 0.5, 0.7])

plt.ylabel("Spearman", fontsize=22)
plt.xticks(x, labels, fontsize=20)

# 显示顶部和右边框线
ax = plt.gca()
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

plt.tick_params(axis="both", which="major", labelsize=20, width=2)

# 在误差棒顶端上方标注均值
for bar, mean_val, std_val in zip(bars, means, stds):
    top = mean_val + std_val   # 误差棒顶端
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        top + 0.02,  # 稍微再往上移，避免遮挡误差棒
        f"{mean_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=16
    )

plt.tight_layout()

out_png = "plot_ESM2_ESMC_ProtT5_spearman.png"
plt.savefig(out_png, bbox_inches="tight", dpi=900)
print(f"已保存图像：{out_png}")
plt.show()
