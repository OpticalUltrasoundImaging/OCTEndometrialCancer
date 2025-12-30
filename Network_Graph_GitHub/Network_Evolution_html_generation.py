"""
Legend: ghost legend from the very beginning (init only).
Before TEXT: grey nodes. After TEXT: colored/shape nodes + hidden edge.
(DBSCAN layers have been removed; all other functionality preserved.)
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# =================== Config ===================
CSV_PATH = "ec_oct_feature_vector_updated070725_57.csv"
OUT_NAME = "threshold_network_nointerp_no_dbscan.html"

THRESHOLDS = np.round(np.arange(1.02, 0.48, -0.02), 2)  # 至 0.50（含）
SEED = 2068 #568

# 动画时序
FRAME_MS = 400
PAUSE_AFTER_TH = 5      # 0.50 → 文字帧 停顿次数
PAUSE_TRUE_LABEL = 3    # 真标签 → 回归线 前的停顿次数
PAUSE_FINAL = 5         # 最后停顿
REG_DELAY_FRAMES = 2    # 回归线相对真标签再延迟出现的帧数

# 外观
NODE_SIZE = 18
NODE_OPACITY = 0.85
NODE_GREY = "#A9A9A9"
EDGE_WIDTH = 1.0
EDGE_OPACITY = 1.0
LEGEND_FONT_SIZE = 22
LEGEND_MARKER_SIZE = NODE_SIZE + 10
LEGEND_LINE_WIDTH = 3
FIG_WIDTH = 1400
FIG_HEIGHT = 1000
LEGEND_AREA = 0.2  # 右侧为 legend 预留宽度

# =================== Data ===================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("__", "_")

node_labels_full: Dict[str, int] = df.groupby("Patient_ID")["Label"].max().to_dict()
feature_cols = [c for c in df.columns if c.startswith("Feature_")]
if not feature_cols:
    raise ValueError("No 'Feature_*' columns found in CSV.")
X_df = df.set_index("Patient_ID")[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)
S = cosine_similarity(X_scaled)
S = (S + 1.0) / 2.0

patient_to_indices: Dict[str, List[int]] = defaultdict(list)
for idx, pid in enumerate(X_df.index):
    patient_to_indices[pid].append(idx)
unique_patients: List[str] = list(patient_to_indices.keys())

patient_sim: Dict[Tuple[str, str], float] = {}
for i, p1 in enumerate(unique_patients):
    idxs1 = patient_to_indices[p1]
    for j in range(i + 1, len(unique_patients)):
        p2 = unique_patients[j]
        idxs2 = patient_to_indices[p2]
        vals = [float(S[a, b]) for a in idxs1 for b in idxs2]
        if vals:
            patient_sim[(p1, p2)] = float(np.mean(vals))

# =================== Helpers ===================

def build_graph(th: float) -> nx.Graph:
    G = nx.Graph()
    for pid in unique_patients:
        G.add_node(pid, Label=int(node_labels_full.get(pid, -1)))
    for (a, b), sim in patient_sim.items():
        if sim >= th:
            G.add_edge(a, b, weight=sim)
    return G


def spring(G: nx.Graph, seed: int, pos_init: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Tuple[float, float]]:
    pos = nx.spring_layout(G, dim=2, seed=seed, pos=pos_init)
    return {k: (float(v[0]), float(v[1])) for k, v in pos.items()}


def edge_xy(G: nx.Graph, pos: Dict[str, Tuple[float, float]]):
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    return ex, ey


def _line_segment_within(xmin, xmax, ymin, ymax, m, b):
    pts = []
    y1 = m * xmin + b
    if ymin <= y1 <= ymax: pts.append((xmin, y1))
    y2 = m * xmax + b
    if ymin <= y2 <= ymax: pts.append((xmax, y2))
    if abs(m) > 1e-12:
        x3 = (ymin - b) / m
        if xmin <= x3 <= xmax: pts.append((x3, ymin))
        x4 = (ymax - b) / m
        if xmin <= x4 <= xmax: pts.append((x4, ymax))
    pts_unique = []
    for p in pts:
        if p not in pts_unique:
            pts_unique.append(p)
    if len(pts_unique) >= 2:
        return [pts_unique[0][0], pts_unique[1][0]], [pts_unique[0][1], pts_unique[1][1]]
    return [xmin, xmax], [m * xmin + b, m * xmax + b]

# =================== Layouts for view locking ===================
# 1) 用 TH=1.00 的布局锁定坐标范围
G_100 = build_graph(1.00)
pos_100 = spring(G_100, SEED, pos_init=None)

xs0 = [pos_100[n][0] for n in G_100.nodes()]
ys0 = [pos_100[n][1] for n in G_100.nodes()]
if len(xs0) == 0: xs0, ys0 = [0.0], [0.0]
margin = 0.10
XRANGE = (min(xs0) - margin, max(xs0) + margin)
YRANGE = (min(ys0) - margin, max(ys0) + margin)

# 2) 用 TH=0.50 的布局作为颜色/形状展示的基准
G_05 = build_graph(0.50)
pos_05 = spring(G_05, SEED, pos_init=None)

ordered_nodes = list(G_05.nodes())
node_positions_arr = np.array([pos_05[n] for n in ordered_nodes])
node_labels_arr = [int(G_05.nodes[n]["Label"]) for n in ordered_nodes]
# Initial 'unit circle' positions for frame 0 (no links)
_angles = np.linspace(0.0, 2*np.pi, len(ordered_nodes), endpoint=False)
circle_positions_arr = np.column_stack([np.cos(_angles), np.sin(_angles)])

# =================== Regression line (precompute; fixed view) ===================
REG_SLOPE = -1.845440094919 # 1.513736721029
REG_INTERCEPT = -0.727469277247 #0.704491632153

slope_r = float(np.round(REG_SLOPE, 2))
intercept_r = float(np.round(REG_INTERCEPT, 2))
sign = '+' if intercept_r >= 0 else '−'
abs_b = abs(intercept_r)
eq_text = f"y = {slope_r}x {sign} {abs_b}"

reg_x, reg_y = _line_segment_within(XRANGE[0], XRANGE[1], YRANGE[0], YRANGE[1], REG_SLOPE, REG_INTERCEPT)

# =================== Groups & ghosts ===================

def idx_where(lbls: List[int], cond) -> List[int]:
    return [i for i, l in enumerate(lbls) if cond(l)]

idx_pre  = idx_where(node_labels_arr, lambda l: l == 0)
idx_post = idx_where(node_labels_arr, lambda l: l == 1)
idx_ein  = idx_where(node_labels_arr, lambda l: l in (2, 3))
idx_ec   = idx_where(node_labels_arr, lambda l: l in (4, 5))


def make_group_trace(idxs: List[int], *, marker_code: str, color: str, filled: bool, name: str, visible: bool):
    symbol_map = {'o': 'circle', '^': 'triangle-up', 'x': 'x'}
    symbol = symbol_map[marker_code]
    xs = [node_positions_arr[i, 0] for i in idxs]
    ys = [node_positions_arr[i, 1] for i in idxs]

    if marker_code == 'o':
        marker = dict(size=NODE_SIZE, symbol=symbol,
                      color=(color if filled else "rgba(0,0,0,0)"),
                      line=dict(color='black', width=2.0), opacity=NODE_OPACITY)
    elif marker_code == '^':
        marker = dict(size=NODE_SIZE, symbol=symbol,
                      color=(color if filled else "rgba(0,0,0,0)"),
                      line=dict(color=color, width=1.5), opacity=NODE_OPACITY)
    else:  # 'x'
        marker = dict(size=NODE_SIZE, symbol='x',
                      color=color, line=dict(width=0), opacity=NODE_OPACITY)

    return go.Scatter(x=xs, y=ys, mode="markers", marker=marker,
                      name=name, hoverinfo="skip", showlegend=False, visible=visible)


def ghost_legend_traces(eq_text: str) -> List[go.Scatter]:
    """Always-visible legend items (do not change across frames)."""
    return [
        go.Scatter(x=[None], y=[None], mode='markers',
                   name="Normal/Benign Endometrium\n(pre-menopause)",
                   marker=dict(size=LEGEND_MARKER_SIZE, symbol='circle',
                               color='black', line=dict(color='black', width=2.5)),
                   showlegend=True, visible=True),
        go.Scatter(x=[None], y=[None], mode='markers',
                   name="Normal/Benign Endometrium\n(post-menopause)",
                   marker=dict(size=LEGEND_MARKER_SIZE, symbol='circle',
                               color='rgba(0,0,0,0)', line=dict(color='black', width=2.5)),
                   showlegend=True, visible=True),
        go.Scatter(x=[None], y=[None], mode='markers',
                   name="EIN",
                   marker=dict(size=LEGEND_MARKER_SIZE, symbol='triangle-up',
                               color='rgba(0,0,0,0)', line=dict(color='red', width=2.0)),
                   showlegend=True, visible=True),
        go.Scatter(x=[None], y=[None], mode='markers',
                   name="EIN/EC",
                   marker=dict(size=LEGEND_MARKER_SIZE, symbol='x', color='red'),
                   showlegend=True, visible=True),
        # 线性回归分割线（legend 专用）
        go.Scatter(x=[None], y=[None], mode='lines',
                   name=f"Linear separator: {eq_text}",
                   line=dict(color='green', width=LEGEND_LINE_WIDTH, dash='dash'),
                   showlegend=True, visible=True),
    ]

# =================== Threshold layout snapshots ===================
pos_prev: Optional[Dict[str, Tuple[float, float]]] = pos_05
th_states = []  # (ex, ey, nx, ny)
for th in THRESHOLDS:
    G = build_graph(float(th))
    if G.number_of_edges() == 0:
        continue
    pos = pos_05 if abs(th - 0.50) < 1e-6 else spring(G, SEED, pos_init=pos_prev)
    pos_prev = pos
    ex, ey = edge_xy(G, pos)
    nx_, ny_ = zip(*[pos[n] for n in G.nodes()]) if len(G.nodes()) else ([], [])
    th_states.append((ex, ey, list(nx_), list(ny_)))
if not th_states:
    raise RuntimeError("No frames built — check data or thresholds.")

# =================== Init data (固定顺序) ===================
# 索引: 0 主边, 1 隐藏边, 2 灰节点, 3 文字,
# 4..7 四组真实节点, 8 回归线, 其后为幽灵 legend（含回归线）
init_edges_main = go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color=f"rgba(128,128,128,{EDGE_OPACITY})", width=EDGE_WIDTH),
                             hoverinfo="skip", showlegend=False, visible=False)
init_edges_hidden = go.Scatter(x=[], y=[], mode="lines",
                               line=dict(color="rgba(0,0,0,0)", width=EDGE_WIDTH),
                               hoverinfo="skip", showlegend=False, visible=False)
init_grey_nodes = go.Scatter(x=circle_positions_arr[:,0], y=circle_positions_arr[:,1], mode="markers",
                             marker=dict(size=NODE_SIZE, color=NODE_GREY, opacity=NODE_OPACITY, line=dict(width=0)),
                             hoverinfo="skip", showlegend=False, visible=True)
init_text = go.Scatter(x=[], y=[], mode="text", text=[],
                       textfont=dict(size=30, color="black"),
                       hoverinfo="skip", showlegend=False, visible=False)

# 真实分组节点（初始隐藏）
init_pre  = make_group_trace(idx_pre,  marker_code='o', color='black', filled=True,  name="Normal/Benign Endometrium\n(pre-menopause)", visible=False)
init_post = make_group_trace(idx_post, marker_code='o', color='black', filled=False, name="Normal/Benign Endometrium\n(post-menopause)", visible=False)
init_ein  = make_group_trace(idx_ein,  marker_code='^', color='red',   filled=False, name="EIN", visible=False)
init_ec   = make_group_trace(idx_ec,   marker_code='x', color='red',   filled=True,  name="EIN/EC", visible=False)

# 回归线（初始隐藏；延迟后显示） —— 绿色
init_reg_line = go.Scatter(
    x=reg_x, y=reg_y, mode="lines",
    line=dict(color="green", width=2, dash="dash"),
    hoverinfo="skip", showlegend=False, visible=False
)

legend_ghosts = ghost_legend_traces(eq_text)

init_data = [init_edges_main, init_edges_hidden, init_grey_nodes, init_text,
             init_pre, init_post, init_ein, init_ec,
             init_reg_line] + legend_ghosts

# =================== Frame builders（不更新幽灵 legend） ===================

def build_init_circle_frame():
    """First frame: no edges, nodes on a unit circle, everything else hidden."""
    data = []
    # 0: main edges (hidden)
    data.append(go.Scatter(x=[], y=[], mode="lines",
                           line=dict(color=f"rgba(128,128,128,{EDGE_OPACITY})", width=EDGE_WIDTH),
                           hoverinfo="skip", showlegend=False, visible=False))
    # 1: hidden edges (hidden)
    data.append(go.Scatter(x=[], y=[], mode="lines",
                           line=dict(color="rgba(0,0,0,0)", width=EDGE_WIDTH),
                           hoverinfo="skip", showlegend=False, visible=False))
    # 2: grey nodes on unit circle
    data.append(go.Scatter(x=circle_positions_arr[:,0], y=circle_positions_arr[:,1], mode="markers",
                           marker=dict(size=NODE_SIZE, color=NODE_GREY, opacity=NODE_OPACITY, line=dict(width=0)),
                           hoverinfo="skip", showlegend=False, visible=True))
    # 3: text (hidden)
    data.append(go.Scatter(x=[], y=[], mode="text", text=[],
                           textfont=dict(size=30, color="black"),
                           hoverinfo="skip", showlegend=False, visible=False))
    # 4..7 groups (hidden)
    data.append(make_group_trace(idx_pre,  marker_code='o', color='black', filled=True,  name="Benign Endometrium(pre-menopause)", visible=False))
    data.append(make_group_trace(idx_post, marker_code='o', color='black', filled=False, name="Benign Endometrium(post-menopause)", visible=False))
    data.append(make_group_trace(idx_ein,  marker_code='^', color='red',   filled=False, name="EIN", visible=False))
    data.append(make_group_trace(idx_ec,   marker_code='x', color='red',   filled=True,  name="EIN/EC", visible=False))
    # 8: regression line (hidden)
    data.append(go.Scatter(x=reg_x, y=reg_y, mode="lines",
                           line=dict(color="green", width=2, dash="dash"),
                           hoverinfo="skip", showlegend=False, visible=False))
    return data


def build_threshold_frame(ex, ey, nx_, ny_):
    data = []
    data.append(go.Scatter(x=ex, y=ey, mode="lines",
                           line=dict(color=f"rgba(128,128,128,{EDGE_OPACITY})", width=EDGE_WIDTH),
                           hoverinfo="skip", showlegend=False, visible=True))   # 0
    data.append(go.Scatter(x=[], y=[], mode="lines",
                           line=dict(color="rgba(0,0,0,0)", width=EDGE_WIDTH),
                           hoverinfo="skip", showlegend=False, visible=False))  # 1
    data.append(go.Scatter(x=nx_, y=ny_, mode="markers",
                           marker=dict(size=NODE_SIZE, color=NODE_GREY, opacity=NODE_OPACITY, line=dict(width=0)),
                           hoverinfo="skip", showlegend=False, visible=True))   # 2
    data.append(go.Scatter(x=[], y=[], mode="text", text=[],
                           textfont=dict(size=30, color="black"),
                           hoverinfo="skip", showlegend=False, visible=False))  # 3
    # 四组真实节点（隐藏）
    data.append(make_group_trace(idx_pre,  marker_code='o', color='black', filled=True,  name="Normal/Benign Endometrium\n(pre-menopause)", visible=False))
    data.append(make_group_trace(idx_post, marker_code='o', color='black', filled=False, name="Normal/Benign Endometrium\n(post-menopause)", visible=False))
    data.append(make_group_trace(idx_ein,  marker_code='^', color='red',   filled=False, name="EIN", visible=False))
    data.append(make_group_trace(idx_ec,   marker_code='x', color='red',   filled=True,  name="EIN/EC", visible=False))
    # 回归线（隐藏）
    data.append(go.Scatter(x=reg_x, y=reg_y, mode="lines",
                           line=dict(color="green", width=2, dash="dash"),
                           hoverinfo="skip", showlegend=False, visible=False))
    return data


def build_text_frame(ex, ey, nx_, ny_):
    # 基于阈值帧的骨架
    data = build_threshold_frame(ex, ey, nx_, ny_)
    # —— 去掉边（文字帧不显示边） ——
    data[0] = go.Scatter(x=[], y=[], mode="lines",
                         line=dict(color=f"rgba(128,128,128,{EDGE_OPACITY})", width=EDGE_WIDTH),
                         hoverinfo="skip", showlegend=False, visible=False)
    data[1] = go.Scatter(x=[], y=[], mode="lines",
                         line=dict(color="rgba(0,0,0,0)", width=EDGE_WIDTH),
                         hoverinfo="skip", showlegend=False, visible=False)
    # 居中提示文字
    cx = 0.5 * (XRANGE[0] + XRANGE[1])
    cy = 0.5 * (YRANGE[0] + YRANGE[1])
    data[3] = go.Scatter(x=[cx], y=[cy], mode="text",
                         text=["<b>Show True Labels</b>"],
                         textfont=dict(size=30, color="black"),
                         hoverinfo="skip", showlegend=False, visible=True)
    return data


def build_true_labels_frame(ex, ey):
    data = []
    # 边全部隐藏（文字帧之后开始）
    data.append(go.Scatter(x=[], y=[], mode="lines",
                           line=dict(color=f"rgba(128,128,128,{EDGE_OPACITY})", width=EDGE_WIDTH),
                           hoverinfo="skip", showlegend=False, visible=False))  # 0
    data.append(go.Scatter(x=[], y=[], mode="lines",
                           line=dict(color="rgba(0,0,0,0)", width=EDGE_WIDTH),
                           hoverinfo="skip", showlegend=False, visible=False))  # 1
    data.append(go.Scatter(x=[], y=[], mode="markers",
                           marker=dict(size=NODE_SIZE, color=NODE_GREY, opacity=NODE_OPACITY, line=dict(width=0)),
                           hoverinfo="skip", showlegend=False, visible=False))  # 2
    data.append(go.Scatter(x=[], y=[], mode="text", text=[],
                           textfont=dict(size=30, color="black"),
                           hoverinfo="skip", showlegend=False, visible=False))  # 3
    # 四组真实节点（打开）
    data.append(make_group_trace(idx_pre,  marker_code='o', color='black', filled=True,  name="Normal/Benign Endometrium\n(pre-menopause)", visible=True))
    data.append(make_group_trace(idx_post, marker_code='o', color='black', filled=False, name="Normal/Benign Endometrium\n(post-menopause)", visible=True))
    data.append(make_group_trace(idx_ein,  marker_code='^', color='red',   filled=False, name="EIN", visible=True))
    data.append(make_group_trace(idx_ec,   marker_code='x', color='red',   filled=True,  name="EIN/EC", visible=True))
    # 回归线（隐藏）
    data.append(go.Scatter(x=reg_x, y=reg_y, mode="lines",
                           line=dict(color="green", width=2, dash="dash"),
                           hoverinfo="skip", showlegend=False, visible=False))
    return data


def build_labels_with_reg_frame(ex, ey):
    data = build_true_labels_frame(ex, ey)
    # 打开回归线
    data[8] = go.Scatter(x=reg_x, y=reg_y, mode="lines",
                         line=dict(color="green", width=2, dash="dash"),
                         hoverinfo="skip", showlegend=False, visible=True)
    return data

# =================== Build frames ===================
frames: List[go.Frame] = []

# 0) 初始帧：单位圆、无连边
frames.append(go.Frame(name="init_circle", data=build_init_circle_frame()))

# 阈值演化帧
for (ex, ey, nx_, ny_) in th_states:
    frames.append(go.Frame(data=build_threshold_frame(ex, ey, nx_, ny_)))

# 文字帧 + 停顿（文字帧也无边）
ex, ey, nx_, ny_ = th_states[-1]
frames.append(go.Frame(name="show_label_hint", data=build_text_frame(ex, ey, nx_, ny_)))
for i in range(max(PAUSE_AFTER_TH - 1, 0)):
    frames.append(go.Frame(name=f"pause_text_{i}", data=build_text_frame(ex, ey, nx_, ny_)))

# 真标签帧 + 停顿（边隐藏）
frames.append(go.Frame(name="true_labels", data=build_true_labels_frame(ex, ey)))
for i in range(PAUSE_TRUE_LABEL):
    frames.append(go.Frame(name=f"pause_true_{i}", data=build_true_labels_frame(ex, ey)))

# 额外增加 REG_DELAY_FRAMES 个“无回归线”的停顿帧
for i in range(REG_DELAY_FRAMES):
    frames.append(go.Frame(name=f"pause_before_reg_{i+1}", data=build_true_labels_frame(ex, ey)))

# 然后一帧：显示回归线
frames.append(go.Frame(name="regression_line", data=build_labels_with_reg_frame(ex, ey)))

# 最后停顿（基于“已显示回归线”的最后一帧）
last_data = frames[-1].data
for i in range(PAUSE_FINAL):
    frames.append(go.Frame(name=f"pause_final_{i}", data=last_data))

# =================== Figure ===================
fig = go.Figure(
    data=init_data,
    frames=frames,
    layout=go.Layout(
        width=FIG_WIDTH, height=FIG_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top', y=1.0,
            xanchor='left', x=1.0 - LEGEND_AREA + 0.01,
            bgcolor='rgba(255,255,255,0.0)',
            font=dict(size=LEGEND_FONT_SIZE),
            traceorder='normal'
        ),
        xaxis=dict(
            domain=[0.0, 1.0 - LEGEND_AREA],
            range=[XRANGE[0], XRANGE[1]],
            constrain='domain', scaleratio=1,
            showgrid=False, zeroline=False
        ),
        yaxis=dict(
            range=[YRANGE[0], YRANGE[1]],
            scaleanchor='x', scaleratio=1,
            showgrid=False, zeroline=False
        ),
        transition=dict(duration=0),
        updatemenus=[], sliders=[],
        margin=dict(l=10, r=10, b=10, t=40),
        paper_bgcolor='white', plot_bgcolor='white'
    )
)

# =================== Export HTML ===================
ANIM_OPTS = dict(
    frame=dict(duration=FRAME_MS, redraw=True),
    transition=dict(duration=0),
    fromcurrent=True,
    mode="immediate",
)

plotly_plot(
    fig,
    filename=OUT_NAME,
    auto_play=True,
    animation_opts=ANIM_OPTS,
    include_plotlyjs="cdn",
    config=dict(responsive=False)
)

print("✅ HTML generated:", OUT_NAME)
