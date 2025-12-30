# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, roc_auc_score


# ================= 配置 =================
CSV_PATH = "patient_layout_coords.csv"   # 输入文件: Plot_ID,x,y,Label
C = 1                               # Logistic 回归正则强度 (越大越弱化正则)
THRESHOLD_MODE = "acc"                # "fixed" | "acc" | "youden"
FIXED_THRESHOLD = 0.5                    # 仅当 THRESHOLD_MODE="fixed" 时使用
SAVE_FIG = "logistic_boundary.png"       # 输出图文件名
USE_EQUAL_AXIS = True                    # 是否等比例坐标

# ================ 读取 & 映射 ================
df = pd.read_csv(CSV_PATH)
for col in ("x", "y", "Label"):
    if col not in df.columns:
        raise ValueError(f"CSV 必须包含列: Plot_ID,x,y,Label（缺少 {col}）")

X = df[["x", "y"]].to_numpy(float)
x = X[:, 0]; y = X[:, 1]
labels_raw = df["Label"].to_numpy(int)

# 按要求的二值化：0/1 -> 0（Benign），2/3/4/5 -> 1（Malignant）
y_true = (labels_raw >= 2).astype(int)

# ================ 训练逻辑回归 ================
clf = LogisticRegression(C=C, solver="lbfgs", max_iter=10000)
clf.fit(X, y_true)

# 预测概率 (阳性=1 的概率)
proba = clf.predict_proba(X)[:, 1]

# ================ 阈值选择 ================
def pick_threshold(y_true, proba, mode="youden", fixed_t=0.5):
    if mode == "fixed":
        return float(fixed_t)
    # 候选阈值：ROC 曲线阈值（覆盖所有可能断点）
    fpr, tpr, thr = roc_curve(y_true, proba)
    if mode == "youden":
        # 最大化 Youden's J = TPR - FPR
        i = np.argmax(tpr - fpr)
        return float(thr[i])
    elif mode == "acc":
        # 用相邻唯一概率的"中点"作为候选阈值，避免恰好等于某个样本概率
        p = np.sort(np.unique(proba))
        # 边界情形：所有概率可能相同
        if p.size == 1:
            cand = np.array([max(1e-6, min(1-1e-6, float(p[0])))])
        else:
            mids = (p[:-1] + p[1:]) / 2.0
            # 在两侧补一个外延中点，并夹紧到 (0,1)
            left  = max(1e-6, float(p[0] / 2.0))
            right = min(1-1e-6, float((1.0 + p[-1]) / 2.0))
            cand = np.r_[left, mids, right]

        acc_best, t_best = -1, 0.5
        for t in cand:
            pred = (proba >= t).astype(int)
            acc = accuracy_score(y_true, pred)
            if acc > acc_best:
                acc_best, t_best = acc, float(t)
        return t_best
    else:
        raise ValueError("THRESHOLD_MODE 必须是 'fixed' | 'acc' | 'youden'")

t_star = pick_threshold(y_true, proba, THRESHOLD_MODE, FIXED_THRESHOLD)

# ================ 计算指标、边界直线参数 ================
def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return sens, spec, acc, (tn, fp, fn, tp)

y_pred = (proba >= t_star).astype(int)
sens, spec, acc, cm = metrics(y_true, y_pred)

# 决策边界： w1*x + w2*y + b0 = logit(t)  ⇒  y = -(w1/w2)*x + (logit(t) - b0)/w2
w1, w2 = clf.coef_[0]
b0 = clf.intercept_[0]
eps = 1e-12
def logit(p): 
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p/(1-p))

if abs(w2) > eps:
    a = -w1 / w2
    b = (logit(t_star) - b0) / w2
    vertical = False
else:
    # 退化为竖线 x = (logit(t) - b0) / w1
    x0 = (logit(t_star) - b0) / w1
    a = None; b = None
    vertical = True

print("=== Logistic Regression ===")
print(f"C = {C}")
print(f"Threshold mode = {THRESHOLD_MODE}, t* = {t_star:.6f}")
print(f"Metrics:  ACC={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, CM(tn,fp,fn,tp)={cm}")
if not vertical:
    print(f"Decision boundary:  y = a*x + b  with  a={a:.12f}, b={b:.12f}")
else:
    print(f"Decision boundary:  vertical line  x = {x0:.12f}")

# ================ 绘图 ================
plt.figure(figsize=(8,6))

# 按原始 Label 的样式（与你之前一致）
marker_map = {0:('o',True), 1:('o',False), 2:('^',False), 3:('^',False), 4:('x',True), 5:('x',True)}
color_map  = {0:'black',   1:'black',     2:'red',      3:'red',       4:'red',     5:'red'}

legend_handles = {}
for lbl in sorted(np.unique(labels_raw)):
    m = labels_raw == lbl
    mk, filled = marker_map.get(int(lbl), ('o', False))
    color = color_map.get(int(lbl), 'gray')
    if mk == 'x':
        sc = plt.scatter(x[m], y[m], marker='x', s=80, color=color, linewidths=2)
    else:
        edge = 'black' if mk == 'o' else color
        face = color if filled else 'none'
        sc = plt.scatter(x[m], y[m], marker=mk, s=80, edgecolors=edge, facecolors=face, linewidths=1.8)
    legend_handles[f"Label {lbl}"] = sc

ax = plt.gca()
xlim = ax.get_xlim(); ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 400)

if not vertical:
    yy = a * xx + b
    (line_plot,) = plt.plot(xx, yy, '--', lw=2.0, color='tab:green',
                            label=(f"Logistic boundary (t={t_star:.3f})\n"
                                   f"$y={a:.2f}x{('+' if b>=0 else '')}{b:.2f}$"))
else:
    (line_plot,) = plt.plot([x0, x0], [ylim[0], ylim[1]], '--', lw=2.0, color='tab:green',
                            label=f"Logistic boundary (t={t_star:.3f})\n$x={x0:.2f}$")

# 恢复坐标范围，外观
ax.set_xlim(xlim); ax.set_ylim(ylim)
if USE_EQUAL_AXIS:
    plt.axis('equal')
plt.xlabel("x"); plt.ylabel("y")
plt.title(f"Logistic Regression (C={C}, threshold={THRESHOLD_MODE})")
handles = list(legend_handles.values()) + [line_plot]
labels = list(legend_handles.keys()) + [line_plot.get_label()]
plt.legend(handles, labels, loc='upper right', frameon=True, framealpha=0.0)
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=300)
plt.show()


# ================ 画 ROC 曲线 ================
from sklearn.metrics import roc_auc_score  # 确保已导入

fpr, tpr, _ = roc_curve(y_true, proba)
auc = roc_auc_score(y_true, proba)

plt.figure(figsize=(6, 5))
ax_roc = plt.gca()

# 黑色 ROC 曲线（在 legend 里显示）
ax_roc.plot(
    fpr, tpr,
    lw=2,
    color='black',
    label=f"ROC Curve(AUC = {auc:.3f})"
)

# random 参考线：显示在图里，但不进 legend
ax_roc.plot(
    [0, 1], [0, 1],
    linestyle='--',
    color='0.7',
    lw=1.5,
    label="_nolegend_"   # 这个 label 会被 legend 忽略
)

# 轴范围改成 -0.2 ~ 1.2
ax_roc.set_xlim(-0.04, 1.04)
ax_roc.set_ylim(-0.04, 1.04)

ax_roc.set_xlabel("1-Specificity", fontsize=16)
ax_roc.set_ylabel("Sensitivity", fontsize=16)
# ax_roc.set_title("ROC Curve")

# 只显示 ROC 曲线这一条 legend
ax_roc.legend(loc="lower right")

# 去掉右边和上边边框
ax_roc.spines['right'].set_visible(False)
ax_roc.spines['top'].set_visible(False)

ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("logistic_roc.png", dpi=300)
plt.show()

# ================ 画 Confusion Matrix ================
# y_true: 0=Benign, 1=EC or EC+EIN
class_names = ["Normal/Benign", "EIN/EC"]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# 各种 rate
TNR = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # true negative rate
FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false positive rate
FNR = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # false negative rate
TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # true positive rate

# 按 TN/FP/FN/TP 对应的 rate 排成矩阵
rate_mat = np.array([
    [TNR, FPR],
    [FNR, TPR]
])

plt.figure(figsize=(5, 4))
ax_cm = plt.gca()

# 颜色可以继续用 Blues，也可以换别的
im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')

# ax_cm.set_title(f"Confusion Matrix (t = {t_star:.3f})")

tick_marks = np.arange(len(class_names))
# x 轴：label 平行于 x 轴
ax_cm.set_xticks(tick_marks)
ax_cm.set_xticklabels(class_names, rotation=0, fontsize=12)

# y 轴：逆时针 90°，平行于 y 轴
ax_cm.set_yticks(tick_marks)
ax_cm.set_yticklabels(class_names, rotation=90, va="center", fontsize=12)

ax_cm.set_ylabel("True label", rotation=90, fontsize=14)
ax_cm.set_xlabel("Predicted label", fontsize=14)

# 在格子里写上“数量 + 对应 rate 百分比”
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        rate = rate_mat[i, j] * 100.0
        text = f"{count}\n({rate:.1f}%)"
        ax_cm.text(
            j, i, text,
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14
        )

# 去掉 colorbar/scale bar（不再调用 colorbar）
# 去掉右上边框，风格跟 ROC 一致
# ax_cm.spines['right'].set_visible(False)
# ax_cm.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig("logistic_confusion_matrix.png", dpi=300)
plt.show()
