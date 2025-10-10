#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, math, warnings, argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ==================== 路径与超参 ====================

WT_DIR = "/home/data1/BGM/MdrDB_All_align/WT_interaction_result"
MT_DIR = "/home/data1/BGM/MdrDB_All_align/MT_interaction_result"
TSV    = "/home/data1/BGM/MdrDB_All_align/MdrDB_mutation_output.tsv"
OUTDIR = "/home/data1/BGM/MdrDB_All_align/MdrDB_interaction_result"
os.makedirs(OUTDIR, exist_ok=True)

BATCH = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42
N_RUNS = 30  # <<< 运行次数

# 边类型
EDGE_TYPES = ["inter", "sm_intra", "prot_intra"]
EDGE_TYPE_IDX = {e:i for i,e in enumerate(EDGE_TYPES)}

# 元素大类；常见元素细分，其余归为 X
ELEMENT_ORDER = ["C", "N", "O", "S", "P", "H", "METAL", "HALO", "X"]
ELEMENT_IDX = {e:i for i,e in enumerate(ELEMENT_ORDER)}

# 半径图兜底的距离阈值（Å）
R_PROT_PROT = 5.0    # protein-protein
R_SM_SM     = 4.0    # smallmol-smallmol
R_INTER     = 5.0    # protein-smallmol

# ==================== DDP 工具 ====================

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pearsonr_np(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 2 or y.size < 2: return np.nan
    xm = x - x.mean(); ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0: return np.nan
    return float((xm*ym).sum() / denom)

def sanitize_mut(mut: str) -> list:
    if mut is None: return []
    cands = [mut, mut.replace("/", "_"), mut.replace(" ", "")]
    seen = set(); out=[]
    for s in cands:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def find_wt_pair(spdb, sid):
    prefix = f"{spdb}_{sid}"
    nodes = os.path.join(WT_DIR, f"{prefix}_nodes.csv")
    edges = os.path.join(WT_DIR, f"{prefix}_edges.csv")
    if os.path.isfile(nodes) and os.path.isfile(edges):
        return nodes, edges
    pat_nodes = os.path.join(WT_DIR, f"{prefix}*_nodes.csv")
    nodes_list = sorted(glob.glob(pat_nodes))
    if nodes_list:
        n = nodes_list[0]; e = n.replace("_nodes.csv", "_edges.csv")
        if os.path.isfile(e):
            return n, e
    return None, None

def find_mt_pair(spdb, sid, mut):
    """ 使用 glob + 合理优先级匹配突变体图文件 """
    for m in sanitize_mut(mut):
        pat_nodes = os.path.join(MT_DIR, f"{spdb}*{sid}*{m}*_nodes.csv")
        nodes_list = sorted(glob.glob(pat_nodes))
        if nodes_list:
            for n in nodes_list:
                e = n.replace("_nodes.csv", "_edges.csv")
                if os.path.isfile(e):
                    return n, e
    pat_nodes = os.path.join(MT_DIR, f"{spdb}*{sid}*_nodes.csv")
    nodes_list = sorted(glob.glob(pat_nodes))
    if nodes_list:
        for n in nodes_list:
            base = os.path.basename(n)
            if mut and (mut in base or any(m in base for m in sanitize_mut(mut))):
                e = n.replace("_nodes.csv","_edges.csv")
                if os.path.isfile(e):
                    return n, e
        n = nodes_list[0]; e = n.replace("_nodes.csv","_edges.csv")
        if os.path.isfile(e):
            return n, e
    return None, None

# =============== 节点特征与建图函数 ===============

def _infer_element_symbol(row) -> str:
    cand_cols = ["element_symbol", "atom", "label"]
    raw = ""
    for c in cand_cols:
        if c in row and pd.notna(row[c]):
            raw = str(row[c]).strip()
            if raw:
                break
    if not raw:
        return "X"
    raw_up = raw.upper()
    two = raw_up[:2]
    if two in ["ZN","MG","FE","CA","MN","CU","CO","NI","NA","K ", "CL","BR","IO","SE"]:
        return two.strip()
    return raw_up[0]

def _bucket_element(sym: str) -> str:
    s = sym.upper()
    if s in ["C","N","O","S","P","H"]:
        return s
    if s in ["ZN","MG","FE","CA","MN","CU","CO","NI","K","NA","SE"]:
        return "METAL"
    if s in ["CL","BR","I","F"]:
        return "HALO"
    return "X"

def atom_feature(row):
    sym = _infer_element_symbol(row)
    buck = _bucket_element(sym)
    elem_oh = np.zeros(len(ELEMENT_ORDER), dtype=np.float32)
    elem_oh[ELEMENT_IDX[buck]] = 1.0

    elem_field = str(row.get("element","")).strip().lower() if pd.notna(row.get("element","")) else ""
    is_prot = 1.0 if elem_field == "protein" else 0.0
    is_lig  = 1.0 if elem_field == "small_molecule" else 0.0

    atom_name = ""
    for c in ["atom", "label"]:
        if c in row and pd.notna(row[c]):
            atom_name = str(row[c]).strip().upper()
            if atom_name:
                break
    is_backbone = 1.0 if atom_name in ["N","CA","C","O"] else 0.0

    return np.concatenate([elem_oh, [is_prot, is_lig, is_backbone]]).astype(np.float32)

def _radius_edges_with_types(pos: np.ndarray, is_prot_mask: np.ndarray, is_lig_mask: np.ndarray):
    N = pos.shape[0]
    src_idx, dst_idx, edge_types, dists = [], [], [], []
    if N == 0:
        return [0,0], [0,0], ["inter","inter"], [0.0, 0.0]
    for i in range(N):
        pi = pos[i]
        for j in range(i+1, N):
            pj = pos[j]
            d = float(np.linalg.norm(pi - pj))
            if is_prot_mask[i] and is_prot_mask[j]:
                thr = R_PROT_PROT; et = "prot_intra"
            elif is_lig_mask[i] and is_lig_mask[j]:
                thr = R_SM_SM; et = "sm_intra"
            else:
                thr = R_INTER; et = "inter"
            if d <= thr:
                src_idx.extend([i, j]); dst_idx.extend([j, i])
                edge_types.extend([et, et])
                dists.extend([d, d])
    if len(src_idx) == 0:
        src_idx = [0,0]; dst_idx=[0,0]; edge_types=["inter","inter"]; dists=[0.0, 0.0]
    return src_idx, dst_idx, edge_types, dists

def build_graph_from_csv(nodes_csv, edges_csv):
    nodes = pd.read_csv(nodes_csv)
    node_ids = list(nodes["node_id"].astype(str).values)
    id2idx = {nid:i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    feats_basic = np.vstack([atom_feature(r) for _, r in nodes.iterrows()])
    is_prot_mask = feats_basic[:, -3] > 0.5
    is_lig_mask  = feats_basic[:, -2] > 0.5

    if all(k in nodes.columns for k in ["x","y","z"]):
        pos = nodes[["x","y","z"]].values.astype(np.float32)
    else:
        pos = np.zeros((N,3), dtype=np.float32)

    use_radius = False
    if os.path.isfile(edges_csv):
        edges = pd.read_csv(edges_csv)
        if edges is None or len(edges) == 0:
            use_radius = True
    else:
        use_radius = True

    src_idx, dst_idx, et_list, dist_list = [], [], [], []

    if not use_radius:
        for _, r in edges.iterrows():
            u = str(r["src"]); v = str(r["dst"])
            if u not in id2idx or v not in id2idx:
                continue
            ui, vi = id2idx[u], id2idx[v]
            src_idx += [ui, vi]; dst_idx += [vi, ui]
            if "distance" in r and pd.notna(r["distance"]):
                d = float(r["distance"])
            else:
                d = float(np.linalg.norm(pos[ui] - pos[vi])) if N>0 else 0.0
            if "edge_type" in r and pd.notna(r["edge_type"]):
                et = str(r["edge_type"])
            else:
                if is_prot_mask[ui] and is_prot_mask[vi]:
                    et = "prot_intra"
                elif is_lig_mask[ui] and is_lig_mask[vi]:
                    et = "sm_intra"
                else:
                    et = "inter"
            et_list += [et, et]
            dist_list  += [d, d]
    else:
        si, di, ets, ds = _radius_edges_with_types(pos, is_prot_mask, is_lig_mask)
        src_idx, dst_idx, et_list, dist_list = si, di, ets, ds

    if len(src_idx) == 0:
        src_idx = [0,0]; dst_idx=[0,0]; et_list=["inter","inter"]; dist_list=[0.0,0.0]

    deg_total = np.zeros((N, 1), dtype=np.float32)
    deg_types = np.zeros((N, len(EDGE_TYPES)), dtype=np.float32)
    dist_sum  = np.zeros((N, 1), dtype=np.float32)
    dist_cnt  = np.zeros((N, 1), dtype=np.float32)

    for (u, v, et, d) in zip(src_idx, dst_idx, et_list, dist_list):
        deg_total[u, 0] += 1
        et = et if et in EDGE_TYPE_IDX else "inter"
        deg_types[u, EDGE_TYPE_IDX[et]] += 1
        dist_sum[u, 0] += float(d)
        dist_cnt[u, 0] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_dist = dist_sum / np.maximum(dist_cnt, 1.0)

    deg_total_norm = np.log1p(deg_total)
    deg_types_norm = np.log1p(deg_types)
    mean_dist_norm = mean_dist

    feats = np.concatenate([feats_basic, deg_total_norm, deg_types_norm, mean_dist_norm], axis=1)

    x = torch.tensor(feats, dtype=torch.float32)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    dist_arr = np.asarray(dist_list, dtype=np.float32)
    edge_weight = torch.tensor(1.0 / (dist_arr + 1e-5), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    if pos is not None and pos.shape[0] == N:
        data.pos = torch.tensor(pos, dtype=torch.float32)
    return data

# =============== 数据集（成对） ===============

class PairGraphDDG(Dataset):
    def __init__(self, tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        first_col = df.columns[0]  # SAMPLE_ID
        needed = ["SAMPLE_PDB_ID", "MUTATION", "DDG.EXP"]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"TSV 缺少列: {c}")

        self.rows = []
        miss = 0
        for _, r in df.iterrows():
            spdb = r["SAMPLE_PDB_ID"]; sid=r[first_col]; mut=r["MUTATION"]; ddg=r["DDG.EXP"]
            if pd.isna(spdb) or pd.isna(sid) or pd.isna(mut) or pd.isna(ddg):
                continue
            wt_nodes, wt_edges = find_wt_pair(spdb, sid)
            mt_nodes, mt_edges = find_mt_pair(spdb, sid, mut)
            if not wt_nodes or not wt_edges or not mt_nodes or not mt_edges:
                miss += 1; continue
            self.rows.append({
                "SAMPLE_PDB_ID": spdb, "SAMPLE_ID": sid, "MUTATION": mut,
                "True_DDG.EXP": float(ddg),
                "wt_nodes": wt_nodes, "wt_edges": wt_edges,
                "mt_nodes": mt_nodes, "mt_edges": mt_edges
            })
        if len(self.rows)==0:
            raise RuntimeError("未匹配到任何 (WT, MUT) 图对。")
        if miss>0 and get_rank()==0:
            print(f"[提示] 有 {miss} 条记录未成功配对，已跳过。")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wt = build_graph_from_csv(row["wt_nodes"], row["wt_edges"])
        mt = build_graph_from_csv(row["mt_nodes"], row["mt_edges"])
        y  = torch.tensor(row["True_DDG.EXP"], dtype=torch.float32)
        meta = {k: row[k] for k in ["SAMPLE_PDB_ID","SAMPLE_ID","MUTATION"]}
        return wt, mt, y, meta

# =============== 模型（孪生 GCN，使用 edge_weight） ===============

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.proj  = nn.Linear(hid, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.drop(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        g = global_mean_pool(h, batch)
        return self.proj(g)

class SiameseDDG(nn.Module):
    def __init__(self, in_dim, hid=128, out=128):
        super().__init__()
        self.enc = GCNEncoder(in_dim, hid, out)
        self.head = nn.Sequential(
            nn.Linear(out*3, 128), nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

    def encode_batch(self, data):
        return self.enc(data.x, data.edge_index, data.batch, getattr(data, "edge_weight", None))

    def forward(self, batch_wt, batch_mut):
        hw = self.encode_batch(batch_wt)
        hm = self.encode_batch(batch_mut)
        z  = torch.cat([hm, hw, hm - hw], dim=-1)
        return self.head(z).squeeze(-1)

# ======== DDP 辅助 ========

def ddp_all_gather_numpy(np_array: np.ndarray) -> np.ndarray:
    if not is_dist():
        return np_array
    device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
    tens = torch.tensor(np_array, device=device, dtype=torch.float32)
    if tens.ndim == 1:
        tens = tens.unsqueeze(0)
    gather_list = [torch.zeros_like(tens) for _ in range(get_world_size())]
    dist.all_gather(gather_list, tens)
    cat = torch.cat(gather_list, dim=0)
    if cat.shape[0] == 1:
        cat = cat.squeeze(0)
    return cat.detach().cpu().numpy().reshape(-1) if np_array.ndim == 1 else cat.detach().cpu().numpy()

def make_pair_loader(dataset, batch_size, sampler=None, num_workers=2, shuffle=None):
    class _Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, pick):
            self.ds = ds; self.pick = pick
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            wt, mt, y, meta = self.ds[idx]
            return (wt if self.pick=="wt" else mt), y, meta

    if shuffle is None:
        shuffle = (sampler is None)

    wrapper_wt = _Wrapper(dataset, "wt")
    wrapper_mt = _Wrapper(dataset, "mt")

    wt_loader = PyGDataLoader(wrapper_wt, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    mt_loader = PyGDataLoader(wrapper_mt, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)

    def _iter():
        for (wt, y1, m1), (mt, y2, m2) in zip(wt_loader, mt_loader):
            assert len(y1)==len(y2)
            yield wt, mt, y1, m1

    steps = (len(dataset) + batch_size - 1) // batch_size
    return _iter, steps

# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", action="store_true", help="启用分布式DDP训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="由 torchrun 传入")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # 分布式初始化
    if args.dist:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        if get_rank() == 0:
            print(f"[DDP] world_size={get_world_size()}  backend=nccl")
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定一个全局基种子，保证每轮随机种子可复现
    set_seed(SEED)

    # 数据只需读一次；每轮用不同随机种子重新划分
    ds = PairGraphDDG(TSV)

    summary_rows = []  # rank0 汇总

    for idx in range(N_RUNS):
        # --- 为本轮生成并同步随机种子 ---
        if args.dist:
            if get_rank() == 0:
                random_seed = random.randint(1, 9999999)
            else:
                random_seed = 0
            seed_tensor = torch.tensor([random_seed], device=device, dtype=torch.int64)
            if get_rank() == 0:
                pass
            dist.broadcast(seed_tensor, src=0)
            random_seed = int(seed_tensor.item())
        else:
            random_seed = random.randint(1, 9999999)

        if get_rank() == 0:
            print(f"\n========== Run {idx+1:02d}/{N_RUNS} | Random seed: {random_seed} ==========")

        set_seed(random_seed)

        # --- 划分数据 ---
        n_total = len(ds)
        n_val = max(1, int(math.ceil(n_total * VAL_SPLIT)))
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(random_seed)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

        # Sampler
        if args.dist:
            train_sampler = DistributedSampler(train_ds, num_replicas=get_world_size(),
                                              rank=get_rank(), shuffle=True, drop_last=False)
            val_sampler   = DistributedSampler(val_ds,   num_replicas=get_world_size(),
                                              rank=get_rank(), shuffle=False, drop_last=False)
        else:
            train_sampler = None
            val_sampler   = None

        iter_train, _ = make_pair_loader(train_ds, BATCH, sampler=train_sampler,
                                         num_workers=args.num_workers,
                                         shuffle=(train_sampler is None))
        iter_val,   _ = make_pair_loader(val_ds,   BATCH, sampler=val_sampler,
                                         num_workers=args.num_workers,
                                         shuffle=False)
        full_iter, _  = make_pair_loader(ds,       BATCH, sampler=None,
                                         num_workers=args.num_workers,
                                         shuffle=False)

        # ===== 输入维度 =====
        in_dim = (len(ELEMENT_ORDER) + 3) + (1 + len(EDGE_TYPES) + 1)  # 17

        # --- 模型/优化器 ---
        model = SiameseDDG(in_dim=in_dim).to(device)
        if args.dist:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
            )
        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.SmoothL1Loss()

        # 目标标准化（注意使用当前划分的训练集）
        y_train_list = []
        for i_tr in (train_ds.indices if hasattr(train_ds, "indices") else range(len(train_ds))):
            _, _, y_tmp, _ = (ds[i_tr] if hasattr(train_ds, "indices") else train_ds[i_tr])
            y_train_list.append(float(y_tmp))
        y_train = torch.tensor(y_train_list, dtype=torch.float32)
        y_mean, y_std = float(y_train.mean()), float(y_train.std().clamp(min=1e-6))
        if get_rank() == 0:
            print(f"[Info][Run {idx:02d}] y_mean={y_mean:.4f}, y_std={y_std:.4f}")

        def norm_y(t):
            return (t - y_mean) / y_std

        best = float("inf")
        prefix = f"run_{idx:02d}_seed{random_seed}"
        best_path = os.path.join(OUTDIR, f"{prefix}_best.pth")
        csv_out   = os.path.join(OUTDIR, f"{prefix}_pred.csv")
        scatter_out = os.path.join(OUTDIR, f"{prefix}_pearson_scatter.png")

        # --- 训练/验证 ---
        for epoch in range(1, EPOCHS+1):
            if args.dist and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            losses=[]
            for wt, mt, y, _ in iter_train():
                wt, mt, y = wt.to(device), mt.to(device), y.to(device)
                pred = model(wt, mt)
                loss = loss_fn(norm_y(pred), norm_y(y))
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())

            scheduler.step()

            if args.dist and val_sampler is not None:
                dist.barrier()

            model.eval()
            vs_local, ys_local = [], []
            with torch.no_grad():
                for wt, mt, y, _ in iter_val():
                    wt, mt = wt.to(device), mt.to(device)
                    p = model(wt, mt).detach().cpu().numpy()
                    vs_local.append(p)
                    ys_local.append(y.numpy())

            if vs_local:
                vs_local = np.concatenate(vs_local)
                ys_local = np.concatenate(ys_local)
            else:
                vs_local = np.array([], dtype=np.float32)
                ys_local = np.array([], dtype=np.float32)

            vs_all = ddp_all_gather_numpy(vs_local)
            ys_all = ddp_all_gather_numpy(ys_local)

            if get_rank() == 0:
                if vs_all.size > 0:
                    rmse = float(np.sqrt(((vs_all-ys_all)**2).mean()))
                    pr = pearsonr_np(vs_all, ys_all)
                else:
                    rmse = np.nan; pr = np.nan
                print(f"[Run {idx:02d}] Epoch {epoch:03d} | TrainLoss {np.mean(losses):.4f} | ValRMSE {rmse:.4f} | ValPearson {pr:.4f}")
            else:
                rmse = 0.0

            device_cur = torch.device(f"cuda:{args.local_rank}" if args.dist else device)
            rmse_tensor = torch.tensor([rmse], device=device_cur, dtype=torch.float32)
            if args.dist:
                dist.broadcast(rmse_tensor, src=0)
            rmse_val = float(rmse_tensor.item())

            if get_rank() == 0 and not np.isnan(rmse_val) and rmse_val < best:
                best = rmse_val
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, best_path)

            if args.dist:
                dist.barrier()

        if get_rank() == 0 and not os.path.isfile(best_path):
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, best_path)

        if args.dist:
            dist.barrier()

        # --- 全量推理 + 保存本轮输出（仅 rank0）---
        if get_rank() == 0:
            raw_model = SiameseDDG(in_dim=in_dim).to(device)
            raw_model.load_state_dict(torch.load(best_path, map_location=device))
            raw_model.eval()

            preds, trues = [], []
            with torch.no_grad():
                for wt, mt, y, meta in full_iter():
                    wt, mt = wt.to(device), mt.to(device)
                    p = raw_model(wt, mt).cpu().numpy()
                    preds.append(p); trues.append(y.numpy())
            preds = np.concatenate(preds); trues = np.concatenate(trues)
            pr_all = pearsonr_np(preds, trues)
            print(f"[Run {idx:02d}] [ALL] Pearson = {pr_all:.4f}")

            rows=[]
            for i, row in enumerate(ds.rows):
                rows.append({
                    "Run": idx,
                    "Seed": random_seed,
                    "SAMPLE_PDB_ID": row["SAMPLE_PDB_ID"],
                    "SAMPLE_ID": row["SAMPLE_ID"],
                    "MUTATION": row["MUTATION"],
                    "True_DDG.EXP": row["True_DDG.EXP"],
                    "Pred_DDG.EXP": float(preds[i]),
                    "ALL_Pearson": pr_all,
                    "Best_Val_RMSE": best
                })
            out_df = pd.DataFrame(rows)
            out_df.to_csv(csv_out, index=False)
            print(f"[OK] 已保存本轮预测: {csv_out}")

            # ---- scatter ----
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pearson_corr = pr_all  # 全量 Pearson
            mn = float(min(trues.min(), preds.min()))
            mx = float(max(trues.max(), preds.max()))
            k, b = np.polyfit(trues, preds, deg=1)
            xs = np.linspace(mn, mx, 100)

            fig, ax = plt.subplots(figsize=(14, 12))

            # 四边框黑色加粗
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # 散点
            ax.scatter(trues, preds, color='#778ccc', label='Data points', s=400)

            # 理想对角线
            ideal_min = float(min(trues.min(), preds.min()))
            ideal_max = float(max(trues.max(), preds.max()))
            ax.plot([ideal_min, ideal_max], [ideal_min, ideal_max],
                    color='gray', linewidth=2, linestyle='dashed', label='Ideal')

            # 拟合直线
            ax.plot(xs, k * xs + b)

            # 坐标轴与刻度（按你给的风格，可自行改回 [-5,6]等）
            ax.set_xlabel("True ΔΔG", fontsize=46)
            ax.set_ylabel("Predicted ΔΔG", fontsize=46)
            ax.set_xlim(-7, 6)
            ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
            ax.set_xticklabels([-6, -4, -2, 0, 2, 4, 6], fontsize=42)
            ax.set_ylim(-7, 6)
            ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
            ax.set_yticklabels([-6, -4, -2, 0, 2, 4, 6], fontsize=42)

            # 图例与文本
            ax.legend(loc='upper left', fontsize=42)
            ax.text(0.55, 0.08, f'Pearson: {pearson_corr:.2f}',
                    transform=ax.transAxes, fontsize=44)

            # 标题与保存 —— 注意这里用 scatter_out 而不是 SCATTER_OUT
            ax.set_title("True vs Predicted ΔΔG - GCN", fontsize=46)
            plt.tight_layout()
            plt.savefig(scatter_out, dpi=900)
            plt.close(fig)
            print(f"[OK] 已保存散点图: {scatter_out}")


        
            summary_rows.append({"Run": idx, "Seed": random_seed, "Best_Val_RMSE": best, "ALL_Pearson": pr_all})
       

    # --- 汇总表（rank0）---
    if get_rank() == 0 and len(summary_rows)>0:
        summ = pd.DataFrame(summary_rows)
        summ_path = os.path.join(OUTDIR, "summary_30runs.csv")
        summ.to_csv(summ_path, index=False)
        print(f"\n[OK] 已保存 30 次随机划分的汇总: {summ_path}")
        print(summ.describe(include='all'))

    if args.dist:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
