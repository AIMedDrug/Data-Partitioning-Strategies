import os
import glob
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------- 基础读取 ----------------
def read_prot_pdb_firstchain(file_path):
    # 读取 ATOM，优先“第一条链”
    chains = set()
    first_chain_id = None
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = (line[21].strip() or " ")
                chains.add(chain_id)
                if first_chain_id is None:
                    first_chain_id = chain_id
    num_chains = len(chains)
    print(f"PDB 文件中共有 {num_chains} 条链(按ATOM/HETATM统计)")
    atoms = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = (line[21].strip() or " ")
                if num_chains == 1 or chain_id == first_chain_id:
                    atom_name = line[12:16].strip()
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    atoms.append((atom_name, np.array([x, y, z])))
    return atoms

def read_ligand_all_hetatm(file_path):
    # 不限制链：收集所有 HETATM 作为小分子原子
    atoms = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                atoms.append((atom_name, np.array([x, y, z])))
    if len(atoms) == 0:
        logging.warning(f"[警告] 未发现 HETATM（小分子）：{file_path}")
    return atoms

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

# ---------------- 构图 ----------------
def generate_interaction_graph_from_complex(complex_pdb_file,
                                            prot_keep_within=5.0,
                                            inter_cutoff=4.5,
                                            sm_internal_cutoff=1.8,
                                            prot_internal_cutoff=1.6):
    # 同一文件中读取：蛋白(ATOM, 首链) + 小分子(HETATM, 全部)
    protein_atoms = read_prot_pdb_firstchain(complex_pdb_file)
    small_molecule_atoms = read_ligand_all_hetatm(complex_pdb_file)

    G = nx.Graph()
    sm_coords = [coord for _, coord in small_molecule_atoms]

    # 仅保留离配体<=5Å的蛋白原子
    filtered_protein_atoms = []
    for atom_name, p_coord in protein_atoms:
        for sm_coord in sm_coords:
            if calculate_distance(p_coord, sm_coord) < prot_keep_within:
                filtered_protein_atoms.append((atom_name, p_coord))
                break

    # 添加节点
    for idx, (atom_name, coord) in enumerate(small_molecule_atoms):
        G.add_node(f"SM_{idx}", element="small_molecule", coords=coord, label=atom_name)
    for idx, (atom_name, coord) in enumerate(filtered_protein_atoms):
        G.add_node(f"P_{idx}", element="protein", coords=coord, label=atom_name)

    # 跨界边（<=4.5Å）
    edge_added = False
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
    for sm_idx, (_, sm_coord) in enumerate(small_molecule_atoms):
        for p_idx, (_, p_coord) in enumerate(filtered_protein_atoms):
            d = calculate_distance(sm_coord, p_coord)
            if 0 < d < inter_cutoff:
                G.add_edge(f"SM_{sm_idx}", f"P_{p_idx}", distance=d, edge_type="inter")
                edge_added = True
    if not edge_added:
        logging.warning("没有符合条件的原子对，未添加任何跨界边。")

    # 小分子内部近邻（<1.8Å）
    small_molecule_edges = []
    for i in range(len(small_molecule_atoms)):
        for j in range(i + 1, len(small_molecule_atoms)):
            d = calculate_distance(small_molecule_atoms[i][1], small_molecule_atoms[j][1])
            if d < sm_internal_cutoff:
                e = (f"SM_{i}", f"SM_{j}")
                small_molecule_edges.append(e)
                G.add_edge(*e, distance=d, edge_type="sm_intra")

    # 蛋白内部近邻（<1.6Å）
    protein_edges = []
    for i in range(len(filtered_protein_atoms)):
        for j in range(i + 1, len(filtered_protein_atoms)):
            d = calculate_distance(filtered_protein_atoms[i][1], filtered_protein_atoms[j][1])
            if d < prot_internal_cutoff:
                e = (f"P_{i}", f"P_{j}")
                protein_edges.append(e)
                G.add_edge(*e, distance=d, edge_type="prot_intra")

    return G, small_molecule_edges, protein_edges

# ---------------- 导出：节点/边 CSV ----------------
def export_graph_csv(G, out_prefix):
    """
    导出两个文件：
      1) {out_prefix}_nodes.csv : node_id, element, label, x, y, z
      2) {out_prefix}_edges.csv : src, dst, distance, edge_type
    """
    import csv

    nodes_csv = out_prefix + "_nodes.csv"
    edges_csv = out_prefix + "_edges.csv"

    # 节点（固定排序，保证可复现）
    node_items = sorted(G.nodes(data=True), key=lambda x: x[0])
    with open(nodes_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "element", "label", "x", "y", "z"])
        for n, data in node_items:
            x, y, z = map(float, data["coords"])
            w.writerow([n, data.get("element",""), data.get("label",""), x, y, z])

    # 边（固定排序）
    edge_items = sorted(G.edges(data=True), key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))
    with open(edges_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "distance", "edge_type"])
        for u, v, data in edge_items:
            w.writerow([u, v, float(data.get("distance", np.nan)), data.get("edge_type","")])

    print(f"已导出: {nodes_csv}\n已导出: {edges_csv}")

# ---------------- 2D 绘图 ----------------
def draw_interaction_graph_2d(G, small_molecule_edges, protein_edges, output_path):
    plt.figure(figsize=(12, 10))
    small_molecule_nodes = [n for n in G.nodes if G.nodes[n]["element"] == "small_molecule"]
    protein_nodes = [n for n in G.nodes if G.nodes[n]["element"] == "protein"]
    pos = {n: G.nodes[n]["coords"][:2] for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, nodelist=small_molecule_nodes, node_color="blue",
                           label="small_molecule", node_size=300)
    nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, node_color="red",
                           label="protein", node_size=300)

    all_edges = set(G.edges())
    internal_edges = set(small_molecule_edges) | set(protein_edges)
    inter_edges = all_edges - internal_edges
    print(f"Inter-edges: {inter_edges}")

    nx.draw_networkx_edges(G, pos, edgelist=list(inter_edges), edge_color="grey", alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=small_molecule_edges, edge_color="green", width=2, label="Small Molecule Bond")
    nx.draw_networkx_edges(G, pos, edgelist=protein_edges, edge_color="orange", width=1, label="Protein Bond")

    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.legend()
    plt.title("Interaction Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"已保存二维相互作用图到 {output_path}")

# ---------------- 文件名清洗 ----------------
def sanitize_for_filename(s):
    # 将不适合文件名的字符替换为下划线
    if s is None:
        return "NA"
    s = str(s).strip()
    return re.sub(r'[^A-Za-z0-9._+-]+', '_', s) or "NA"

# ---------------- 主流程（突变型） ----------------
def main():
    complex_root = "/home/data1/BGM/MdrDB_All_align/MdrDB_complex_result"
    tsv_file     = "/home/data1/BGM/MdrDB_All_align/MdrDB_mutation_output.tsv"
    output_dir   = "/home/data1/BGM/MdrDB_All_align/MT_interaction_result"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(tsv_file, sep="\t", dtype=str)

    # 必要列检查
    first_col_name = df.columns[0]  # SAMPLE_ID
    if "MUTATION" not in df.columns:
        raise ValueError("TSV 中未找到列名 'MUTATION'，请检查文件。")
    has_pdbid = "SAMPLE_PDB_ID" in df.columns

    # 取 (SAMPLE_ID, MUTATION, SAMPLE_PDB_ID?) 去重
    cols = [first_col_name, "MUTATION"] + (["SAMPLE_PDB_ID"] if has_pdbid else [])
    pairs = df[cols].dropna(subset=[first_col_name, "MUTATION"]).drop_duplicates().values.tolist()

    print(f"待处理样本数：{len(pairs)}")
    for row in pairs:
        if has_pdbid:
            SAMPLE_ID, MUTATION, SAMPLE_PDB_ID = row
        else:
            SAMPLE_ID, MUTATION = row
            SAMPLE_PDB_ID = "NA"

        subdir = os.path.join(complex_root, SAMPLE_ID)
        if not os.path.isdir(subdir):
            print(f"[跳过] 未找到子文件夹：{subdir}")
            continue

        mut_key = sanitize_for_filename(MUTATION)

        # 优先匹配包含 mutation 关键字的 MT_*_complex.pdb
        pattern_precise = os.path.join(subdir, f"MT_*{mut_key}*_complex.pdb")
        mt_candidates = sorted(glob.glob(pattern_precise))

        # 若未匹配到，退而求其次：任意 MT_*_complex.pdb（给出提示）
        if len(mt_candidates) == 0:
            pattern_relaxed = os.path.join(subdir, "MT_*_complex.pdb")
            mt_candidates = sorted(glob.glob(pattern_relaxed))
            if len(mt_candidates) == 0:
                print(f"[跳过] 子文件夹 {subdir} 中未找到 MT_*_complex.pdb")
                continue
            else:
                print(f"[提示] 未找到包含突变关键字({mut_key})的文件，改用第一个：{mt_candidates[0]}")

        complex_pdb = mt_candidates[0]

        # 输出命名：{SAMPLE_PDB_ID}_{SAMPLE_ID}_{MUTATION}_interaction_graph_*.* + *_nodes/edges.csv
        out_prefix      = os.path.join(
            output_dir,
            f"{sanitize_for_filename(SAMPLE_PDB_ID)}_{sanitize_for_filename(SAMPLE_ID)}_{mut_key}"
        )
        output_path_2d  = out_prefix + "_interaction_graph_2d.png"

        print(f"\n=== 开始处理 SAMPLE_ID={SAMPLE_ID}, MUTATION={MUTATION}, SAMPLE_PDB_ID={SAMPLE_PDB_ID} ===")
        print(f"突变复合物PDB: {complex_pdb}")
        print(f"输出二维PNG:  {output_path_2d}")

        try:
            G, sm_edges, protein_edges = generate_interaction_graph_from_complex(complex_pdb)

            # 先导出节点/边列表（供 GCN 使用）
            export_graph_csv(G, out_prefix)

            # 再画二维图
            draw_interaction_graph_2d(G, sm_edges, protein_edges, output_path_2d)

            print(f"[成功] {SAMPLE_ID} | {MUTATION} 处理完成\n")
        except Exception as e:
            print(f"[错误] 处理 {SAMPLE_ID} | {MUTATION} 时出错: {e}\n")

if __name__ == "__main__":
    main()
