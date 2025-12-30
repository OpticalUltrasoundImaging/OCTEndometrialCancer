import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ====== 顶部 imports 处，保留导出需要 ======
import json
import pickle
from pathlib import Path

# ========== 读取和预处理数据 ========== #
# df = pd.read_csv("ec_oct_feature_vector_updated070725.csv")
# df = pd.read_csv("ec_oct_feature_vector_updated070725_tsne2d.csv")
df = pd.read_csv("ec_oct_feature_vector_updated070725_57.csv")

df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("__", "_")

node_ids = df["Patient_ID"].tolist()
df_label_max = df.groupby("Patient_ID")["Label"].max()
node_labels = df_label_max.to_dict()

feature_cols = [col for col in df.columns if col.startswith("Feature_")]
df_features = df.set_index("Patient_ID")[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
similarity_matrix = cosine_similarity(X_scaled)
similarity_matrix = (similarity_matrix + 1) / 2

# 创建导出目录
OUT_DIR = Path("export_graph_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# —— 为了让 CSV 的行列标签不冲突，给同一患者的多行加序号后缀
_cnt = defaultdict(int)
sample_ids = []
for pid in df_features.index:
    _cnt[pid] += 1
    sample_ids.append(f"{pid}#{_cnt[pid]}")

# 保存“样本级”相似度矩阵
sim_df = pd.DataFrame(similarity_matrix, index=sample_ids, columns=sample_ids)
sim_df.to_csv(OUT_DIR / "similarity_matrix_raw.csv", index=True)
np.save(OUT_DIR / "similarity_matrix_raw.npy", similarity_matrix)

# ========== 构建 patient-level 相似度（跨侧均值）并保存 ========== #
patient_to_indices = defaultdict(list)
for idx, pid in enumerate(df_features.index):
    patient_to_indices[pid].append(idx)

unique_patients = list(patient_to_indices.keys())
P = len(unique_patients)
patient_mat = np.full((P, P), np.nan, dtype=float)

for i, pid1 in enumerate(unique_patients):
    patient_mat[i, i] = 1.0  # 自身相似度
    idxs1 = patient_to_indices[pid1]
    for j in range(i + 1, P):
        pid2 = unique_patients[j]
        idxs2 = patient_to_indices[pid2]
        sims = [similarity_matrix[m, n] for m in idxs1 for n in idxs2]
        if len(sims) > 0:
            avg_sim = float(sum(sims) / len(sims))
            patient_mat[i, j] = avg_sim
            patient_mat[j, i] = avg_sim

# 保存“患者级平均”相似度矩阵
patient_sim_df = pd.DataFrame(patient_mat, index=unique_patients, columns=unique_patients)
patient_sim_df.to_csv(OUT_DIR / "patient_similarity_matrix_avg.csv", index=True)
np.save(OUT_DIR / "patient_similarity_matrix_avg.npy", patient_mat)

# （可选）也保存阈值后的邻接矩阵，便于对照你图上的连边
threshold = 0.5  # 用你图中一致的阈值
A = (patient_sim_df.values >= threshold).astype(int)
np.fill_diagonal(A, 0)
adj_df = pd.DataFrame(A, index=unique_patients, columns=unique_patients)
adj_df.to_csv(OUT_DIR / "patient_adjacency_thresholded.csv", index=True)

# ========== 构建 patient-level 相似度矩阵（成对平均） ========== #
patient_to_indices = defaultdict(list)
for idx, pid in enumerate(df_features.index):
    patient_to_indices[pid].append(idx)

unique_patients = list(patient_to_indices.keys())
patient_sim_matrix = dict()
for i, pid1 in enumerate(unique_patients):
    for j in range(i + 1, len(unique_patients)):
        pid2 = unique_patients[j]
        idxs1 = patient_to_indices[pid1]
        idxs2 = patient_to_indices[pid2]
        sims = [similarity_matrix[m, n] for m in idxs1 for n in idxs2]
        if sims:
            avg_sim = sum(sims) / len(sims)
            patient_sim_matrix[(pid1, pid2)] = avg_sim

# ========== 构建图结构并布局（多条边取均值） ========== #
threshold = 0.5

# 1) 先“装袋”：同一对端点 -> 多个候选权重
edge_bag = defaultdict(list)
for (a, b), sim in patient_sim_matrix.items():
    if sim >= threshold and a != b:           # 候选阈值，可改为后置阈值
        u, v = sorted((a, b))                 # 无向图统一键
        edge_bag[(u, v)].append(sim)

# 导出边包统计
with open(OUT_DIR / "edge_bag.pkl", "wb") as f:
    pickle.dump(edge_bag, f)

rows = []
for (u, v), ws in edge_bag.items():
    if not ws:
        continue
    rows.append({
        "u": u,
        "v": v,
        "n": len(ws),
        "mean": float(sum(ws) / len(ws)),
        "min": float(min(ws)),
        "max": float(max(ws)),
        "values_json": json.dumps([float(x) for x in ws]),
    })
pd.DataFrame(rows).to_csv(OUT_DIR / "edge_bag_summary.csv", index=False)

# 2) 建图：对每对端点求均值后写入一条边
G = nx.Graph()
for pid in unique_patients:
    G.add_node(pid, Label=node_labels[pid])

for (u, v), ws in edge_bag.items():
    avg_sim = sum(ws) / len(ws)
    G.add_edge(u, v, weight=avg_sim)

# 导出最终图的边
edges_df = nx.to_pandas_edgelist(G)  # columns: source, target, weight
edges_df.to_csv(OUT_DIR / "graph_edges_after_threshold.csv", index=False)

# 3) 后续布局与取坐标
pos_2d = nx.spring_layout(G, dim=2, seed=2068) #568
node_positions = np.array([pos_2d[node] for node in G.nodes()])
node_ids_list = list(G.nodes())
node_labels_list = [G.nodes[node]["Label"] for node in node_ids_list]

# === 新增：保存每个点的最终 2D 坐标到 CSV ===
coords_rows = []
for node in G.nodes():
    x, y = pos_2d[node]  # spring_layout 的最终坐标
    coords_rows.append({
        "Plot_ID": node,
        "x": float(x),
        "y": float(y),
        "Label": G.nodes[node]["Label"]
    })

coords_df = pd.DataFrame(coords_rows)
coords_df.to_csv("patient_layout_coords.csv", index=False, encoding="utf-8-sig")
print(f"已保存 {len(coords_df)} 个节点坐标到 patient_layout_coords.csv")

# ========== 开始绘图（无 DBSCAN 阴影） ========== #
plt.figure(figsize=(10, 8))

label_color_map = {
    0: ("black", "Normal/Benign Endometrium\n(pre-menopause)"),
    1: ("black", "Normal/Benign Endometrium\n(post-menopause)"),
    2: ("red", "EIN"),
    3: ("red", "EIN"),
    4: ("red", "EIN/EC"),
    5: ("red", "EIN/EC")
}

marker_map = {
    0: ('o', True),     # 实心圆
    1: ('o', False),    # 空心圆
    2: ('^', False),    # 三角
    3: ('^', False),    # 三角
    4: ('x', True),     # X
    5: ('x', True)
}

legend_elements = {}

for i, label in enumerate(node_labels_list):
    x, y = node_positions[i]
    marker, filled = marker_map[label]
    color, label_name = label_color_map[label]

    if marker == 'o':
        plt.scatter(x, y, marker=marker,
                    s=120, edgecolors='black',
                    facecolors=color if filled else 'none',
                    linewidths=2.0, zorder=3.0)
    elif marker == '^':
        plt.scatter(x, y, marker=marker,
                    s=120,
                    facecolors=color if filled else 'none',
                    edgecolors=color,
                    linewidths=1.5, zorder=3.0)
    elif marker == 'x':
        plt.scatter(x, y, marker=marker,
                    s=120, color=color,
                    linewidths=3, zorder=3.0)

    # legend 元素
    if label_name not in legend_elements:
        if marker == 'o':
            legend_elements[label_name] = plt.Line2D([0], [0], marker=marker,
                                                     color='black',
                                                     markerfacecolor=color if filled else 'none',
                                                     markersize=10,
                                                     markeredgewidth=2.0,
                                                     label=label_name,
                                                     linestyle='None')
        elif marker == '^':
            legend_elements[label_name] = plt.Line2D([0], [0],
                                                     marker=marker,
                                                     markerfacecolor=color if filled else 'none',
                                                     markeredgecolor=color,
                                                     markersize=10,
                                                     markeredgewidth=2.0,
                                                     label=label_name,
                                                     linestyle='None')
        else:
            legend_elements[label_name] = plt.Line2D([0], [0], marker=marker,
                                                     color=color,
                                                     markeredgewidth=2.0,
                                                     markersize=10,
                                                     label=label_name,
                                                     linestyle='None')

# 坐标等比
plt.axis('equal')

# 锁定当前坐标范围
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 线性回归分割线（不改变布局）
slope = -1.845440094919
intercept = -0.727469277247
slope_r = float(np.round(slope, 2))
intercept_r = float(np.round(intercept, 2))
sign = '+' if intercept_r >= 0 else '-'
abs_b = abs(intercept_r)

x_line = np.linspace(xlim[0], xlim[1], 500)
y_line = slope * x_line + intercept
mask = (y_line >= ylim[0]) & (y_line <= ylim[1])

label_line = f"Logistic Regression Model \n $y = {slope_r:.2f}\,x {sign} {abs_b:.2f}$"

(reg_line,) = ax.plot(
    x_line[mask], y_line[mask],
    linestyle='--',
    linewidth=2.0,
    color='tab:green',
    zorder=4.0,
    scalex=False,
    scaley=False,
    clip_on=True,
    label=label_line
)

# 恢复坐标范围
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# ➤ 固定 legend 顺序为合并后的四项，并把回归线加入；位置改为“右上角”
ordered_names = [
    "Normal/Benign Endometrium\n(pre-menopause)",
    "Normal/Benign Endometrium\n(post-menopause)",
    "EIN",
    "EIN/EC"
]
ordered_handles = [legend_elements[name] for name in ordered_names] + [reg_line]

# —— 图例：移动到右上角；不再降低 zorder（避免被遮挡）
# leg = plt.legend(handles=ordered_handles,
#                  loc='lower right',   # ← 移到右上角
#                  markerscale=1.2,
#                  ncol=1,
#                  labelspacing=1,
#                  handletextpad=0.4,
#                  borderpad=0.8,
#                  frameon=True,
#                  framealpha=0.0,     # 需要完全透明可保留；若想显示底色可改为 0.8
#                  fancybox=False,
#                  prop={'size': 11.6})

# frame = leg.get_frame()
# frame.set_linewidth(0.0)
# frame.set_edgecolor((0, 0, 0, 0))
# frame.set_facecolor((1, 1, 1, 0))

leg = plt.legend(handles=ordered_handles,
                 loc='lower right',
                 markerscale=1.2,
                 ncol=1,
                 labelspacing=1,
                 handletextpad=0.4,
                 borderpad=0.8,
                 frameon=True,        # 要保留这个 True
                 framealpha=0.9,      # 背景稍微有点不透明
                 fancybox=False,
                 prop={'size': 13})

frame = leg.get_frame()
frame.set_linewidth(1.5)               # 边框线粗一点
frame.set_edgecolor('black')           # 边框线颜色（黑色）
frame.set_facecolor((1, 1, 1, 0.9))    # 白色背景 + 0.9 透明度

# 其他外观
plt.xticks(color='black', fontsize=18)
plt.yticks(color='black', fontsize=18)
plt.gca().set_facecolor("white")
plt.tight_layout()
plt.show()
