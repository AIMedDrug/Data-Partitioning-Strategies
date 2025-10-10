import os
import sys
import shutil
import pandas as pd

# ======= 路径配置（按你的需求可改）=======
tsv_path       = "/home/data1/BGM/MdrDB_All_align/MdrDB_mutation_output.tsv"
structure_root = "/home/data1/BGM/MdrDB_CoreSet_v1.0.2022/structure"
dst_root       = "/home/data1/BGM/MdrDB_All_align/MdrDB_complex_result"

# ======= 可选：当一格里有多个ID时的分隔符集合 =======
SPLITTERS = [",", ";", "|"]

def split_ids(x: str):
    """将可能包含多个ID的一格拆分成列表，并去除空白。"""
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    parts = [s]
    for sp in SPLITTERS:
        if sp in s:
            parts = [p.strip() for p in s.split(sp)]
            break
    return [p for p in parts if p]

def main():
    # 读TSV
    if not os.path.isfile(tsv_path):
        print(f"[错误] 找不到TSV文件：{tsv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)  # 全部按字符串读，避免类型混淆

    # 列名检查
    # 第一列名可能不是"SAMPLE_ID"，但你说“第一列SAMPLE_ID”，这里严格按这两个列名处理
    required_cols = ["SAMPLE_ID", "SAMPLE_PDB_ID"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[错误] TSV中缺少列：{c}", file=sys.stderr)
            print(f"[提示] 实际列名有：{list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # 目标目录
    os.makedirs(dst_root, exist_ok=True)

    # 结构目录：收集一层子文件夹（只匹配第一层目录名）
    if not os.path.isdir(structure_root):
        print(f"[错误] 找不到structure根目录：{structure_root}", file=sys.stderr)
        sys.exit(1)

    subdir_map = {}  # 名称 -> 绝对路径
    for name in os.listdir(structure_root):
        full = os.path.join(structure_root, name)
        if os.path.isdir(full):
            subdir_map[name] = full
    print(f"[信息] structure子文件夹数量：{len(subdir_map)}")

    copied = set()
    miss   = []   # 记录未匹配到的行（以便后续排查）

    for idx, row in df.iterrows():
        sample_id_raw = row.get("SAMPLE_ID", "")
        pdb_id_raw    = row.get("SAMPLE_PDB_ID", "")

        pdb_ids    = split_ids(pdb_id_raw)
        sample_ids = split_ids(sample_id_raw)

        matched_name = None

        # 优先用 SAMPLE_PDB_ID 匹配
        for cand in pdb_ids:
            if cand in subdir_map:
                matched_name = cand
                break

        # 若没匹配到，用 SAMPLE_ID 再试
        if matched_name is None:
            for cand in sample_ids:
                if cand in subdir_map:
                    matched_name = cand
                    break

        if matched_name is None:
            miss.append((idx, sample_id_raw, pdb_id_raw))
            continue

        # 已匹配：复制（去重）
        if matched_name in copied:
            continue

        src_dir = subdir_map[matched_name]
        dst_dir = os.path.join(dst_root, matched_name)

        # 执行复制
        if os.path.exists(dst_dir) and os.listdir(dst_dir):
            # 已存在且非空，认为已复制过/或已有内容，跳过或你也可选择覆盖
            print(f"[跳过] 目标已存在且非空：{dst_dir}")
            copied.add(matched_name)
            continue

        print(f"[复制] {src_dir}  ->  {dst_dir}")
        # Python 3.8+ 支持 dirs_exist_ok
        try:
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        except TypeError:
            # 若是老版本Python不支持dirs_exist_ok，则fallback
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            # 逐项复制
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dst_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)

        copied.add(matched_name)

    print("\n===== 统计 =====")
    print(f"成功复制文件夹数：{len(copied)}")
    print(f"未匹配到的行数：{len(miss)}")
    if miss:
        # 打印前若干个未匹配示例
        print("未匹配示例（前10行）：")
        for i, (idx, sid, pid) in enumerate(miss[:10]):
            print(f"  行{idx}: SAMPLE_ID='{sid}'  SAMPLE_PDB_ID='{pid}'")
        # 如需导出未匹配清单，可解除下面注释
        # import csv
        # miss_csv = os.path.join(dst_root, "not_matched_rows.csv")
        # with open(miss_csv, "w", newline="") as f:
        #     w = csv.writer(f)
        #     w.writerow(["row_index", "SAMPLE_ID", "SAMPLE_PDB_ID"])
        #     w.writerows(miss)
        # print(f"[信息] 未匹配清单已导出：{miss_csv}")

if __name__ == "__main__":
    main()
