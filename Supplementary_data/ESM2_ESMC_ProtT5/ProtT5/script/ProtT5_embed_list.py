import os
import glob
import torch
import numpy as np
import pandas as pd

# ================== 路径配置 ==================
base_dir = "/home/bioinfor6/BGM/fenziduiqi/ProtT5"
input_csv = os.path.join(base_dir, "embed_list.csv")     # 第一列=PDB，第二列=Mutation，X1/X2 空
wild_dir  = os.path.join(base_dir, "ProtT5_Wild_result")   # 野生型 .pt 所在目录
mut_dir   = os.path.join(base_dir, "ProtT5_Mut_result")    # 突变体 .pt 所在目录
output_csv = os.path.join(base_dir, "ProtT5_embed_list.csv")

# ================== 工具函数（针对 ESMC 优化） ==================
def to_str_vec(t):
    return ",".join(map(str, t))

def is_probably_state_dict(obj):
    return isinstance(obj, dict) and sum(1 for k in obj.keys() if "weight" in str(k).lower()) >= 5

def clean_token_matrix(tok: torch.Tensor) -> torch.Tensor:
    """对 token 级表示 [L, d] 做均值；若 L>=3，去掉首尾 token（通常是 BOS/EOS）"""
    if tok.ndim != 2:
        return tok
    L, _ = tok.shape
    if L >= 3:
        return tok[1:-1].mean(dim=0)
    return tok.mean(dim=0)

def _iter_tensors(obj, path="root"):
    if torch.is_tensor(obj):
        yield obj, path
        return
    if isinstance(obj, np.ndarray):
        yield torch.from_numpy(obj), path
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_tensors(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{path}[{i}]")
        return

def extract_esmc_tensor(obj: object) -> torch.Tensor:
    """
    针对 ESM/ESMC 的常见保存格式做优先解析：
      - mean_representations / sequence_embedding(s) / pooled / sequence_repr -> 1D
      - token_representations / representations[layer] -> [L,d] 再均值
      - 其余情况：递归遍历所有张量/数组，选一个最像样本表征的
    """
    # 直接 tensor / ndarray
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)

    # state_dict 直接报错
    if is_probably_state_dict(obj):
        raise ValueError("加载到的对象像是模型 state_dict（权重），不是样本 embedding。")

    # 常见 ESM/ESMC 字典结构
    if isinstance(obj, dict):
        # 1) 直接给出的序列级向量
        for k in ("mean_representations", "sequence_embeddings", "sequence_embedding",
                  "sequence_repr", "pooled", "mean", "avg"):
            if k in obj:
                v = obj[k]
                if torch.is_tensor(v):
                    return v
                if isinstance(v, np.ndarray):
                    return torch.from_numpy(v)

        # 2) token 级矩阵
        #   2.1 token_representations: [L, d] 或 list/tuple
        for k in ("token_representations", "token_embedding", "token_embeddings"):
            if k in obj:
                v = obj[k]
                if torch.is_tensor(v) and v.ndim == 2:
                    return clean_token_matrix(v)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    last = v[-1]
                    if torch.is_tensor(last) and last.ndim == 2:
                        return clean_token_matrix(last)
                    if isinstance(last, np.ndarray) and last.ndim == 2:
                        return clean_token_matrix(torch.from_numpy(last))

        #   2.2 representations: {layer:int -> [L,d]}
        if "representations" in obj and isinstance(obj["representations"], dict) and len(obj["representations"]) > 0:
            reps = obj["representations"]
            try:
                last_layer = sorted(reps.keys())[-1]
                t = reps[last_layer]
                if torch.is_tensor(t) and t.ndim == 2:
                    return clean_token_matrix(t)
                if isinstance(t, np.ndarray) and t.ndim == 2:
                    return clean_token_matrix(torch.from_numpy(t))
            except Exception:
                pass

        # 3) 其他键位的 1D/2D 表征
        for k in ("embedding", "emb", "x", "feat", "hidden_states", "hidden"):
            if k in obj:
                v = obj[k]
                if torch.is_tensor(v):
                    return v if v.ndim == 1 else clean_token_matrix(v) if v.ndim == 2 else v.flatten()
                if isinstance(v, np.ndarray):
                    vt = torch.from_numpy(v)
                    return vt if vt.ndim == 1 else clean_token_matrix(vt) if vt.ndim == 2 else vt.flatten()

    # 回退：递归搜集所有张量，按启发式挑一个
    cands = list(_iter_tensors(obj))
    if not cands:
        raise ValueError("对象内没有可用张量/数组。")

    # 优先 2D token 表征，再 1D
    best = None
    best_score = -1
    for t, p in cands:
        if not torch.is_tensor(t):
            continue
        score = 0
        if t.ndim == 2:
            L, d = t.shape
            if L >= 3 and d <= 65536:
                score += 100
        elif t.ndim == 1:
            score += 70
        # 路径关键词加分
        if any(kw in p.lower() for kw in ["representations", "token", "embedding", "sequence", "pooled", "mean"]):
            score += 15
        if score > best_score:
            best = (t, p)
            best_score = score

    if best is None:
        raise ValueError("未能在对象中找到合适的张量。")
    t, _ = best
    return clean_token_matrix(t) if t.ndim == 2 else (t.flatten() if t.ndim > 1 else t)

def load_pt_vector(pt_path: str) -> str:
    """统一把任意 ESMC 保存的 .pt 解析成 1D 序列向量字符串"""
    obj = torch.load(pt_path, map_location="cpu")
    t = extract_esmc_tensor(obj)
    if t.ndim == 2:
        t = clean_token_matrix(t)
    elif t.ndim > 2:
        t = t.flatten()
    vec = t.detach().cpu().numpy()
    return to_str_vec(vec)

def find_mut_file(pdb_id, mutation):
    pattern = os.path.join(mut_dir, f"{pdb_id}|*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    for path in candidates:
        fname = os.path.basename(path)
        try:
            middle = fname.split("|", 1)[1].rsplit(".pt", 1)[0]
        except Exception:
            continue
        if middle == mutation:
            return path
    return None

# ================== 主流程 ==================
def main():
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到输入 CSV：{input_csv}")
    if not os.path.isdir(wild_dir):
        raise NotADirectoryError(f"找不到野生型目录：{wild_dir}")
    if not os.path.isdir(mut_dir):
        raise NotADirectoryError(f"找不到突变体目录：{mut_dir}")

    df = pd.read_csv(input_csv)

    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：第一列为 PDB 名，第二列列名为 Mutation。")

    pdb_col = df.columns[0]
    mut_col = "Mutation" if "Mutation" in df.columns else df.columns[1]
    if "Mutation" not in df.columns:
        print(f"⚠️ 注意：第二列不是 'Mutation'，而是 '{mut_col}'，将按它作为突变列。")

    if "X1" not in df.columns:
        df["X1"] = ""
    if "X2" not in df.columns:
        df["X2"] = ""

    miss_wild = miss_mut = ok_wild = ok_mut = 0

    for i, row in df.iterrows():
        pdb_id = str(row[pdb_col]).strip()
        mutation = str(row[mut_col]).strip()

        # ------- Wild (X1) -------
        wild_path = os.path.join(wild_dir, f"{pdb_id}.pt")
        if os.path.exists(wild_path):
            try:
                df.at[i, "X1"] = load_pt_vector(wild_path)
                ok_wild += 1
            except Exception as e:
                miss_wild += 1
                print(f"❌ 第 {i} 行 {pdb_id} 加载野生型失败：{e}")
        else:
            miss_wild += 1
            print(f"❌ 第 {i} 行 未找到野生型文件：{wild_path}")

        # ------- Mut (X2) -------
        mut_path = find_mut_file(pdb_id, mutation)
        if mut_path and os.path.exists(mut_path):
            try:
                df.at[i, "X2"] = load_pt_vector(mut_path)
                ok_mut += 1
            except Exception as e:
                miss_mut += 1
                print(f"❌ 第 {i} 行 {pdb_id}|{mutation} 加载突变体失败：{e}")
        else:
            miss_mut += 1
            print(f"❌ 第 {i} 行 未找到突变体精确匹配文件：{pdb_id}|{mutation}.pt")

        if (i + 1) % 50 == 0:
            print(f"…进度 {i+1}/{len(df)}")

    df.to_csv(output_csv, index=False)
    print("\n====================== 完成 ======================")
    print(f"✅ 已输出：{output_csv}")
    print(f"✅ 野生型成功 {ok_wild} 条，缺失/失败 {miss_wild} 条")
    print(f"✅ 突变体成功 {ok_mut} 条，缺失/失败 {miss_mut} 条")

if __name__ == "__main__":
    main()
