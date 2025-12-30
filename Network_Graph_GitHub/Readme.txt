## Network Graph Visualization (2D Plot & Evolution GIF)

This repo supports:

1. **2D plotting of the network graph** (with a regression-based decision boundary), and
2. **GIF generation for the network evolution** (HTML → GIF).

---

### 1) 2D Network Graph Plotting (with Decision Boundary)

**Step 1 — Generate node coordinates**
Run:

* `Network_graph_ploting.py`

This will output:

* `patient_layout_coords.csv`
  (stores the 2D layout coordinates for each patient/node; used by the regression model)

**Step 2 — Fit decision boundary**
Use `patient_layout_coords.csv` to train/fit your regression model and obtain the decision boundary:

[
y = a \cdot x + b
]

**Step 3 — Update coefficients and re-plot**
Open `Network_graph_ploting.py`, replace the existing `a` and `b` with the fitted coefficients, then run it again to generate the final 2D plot with the decision boundary.

---

### 2) Network Evolution GIF (HTML → GIF)

**Step 1 — Generate the evolution HTML**
Run:

* `Network_Evolution_html_generation.py`

This will output:

* an evolution **HTML** file (interactive animation of the network evolution)

**Step 2 — Convert HTML to GIF**
Run:

* `convert_html_2_gif.py`

This will convert the generated HTML animation into a **GIF**.

---

### Outputs Summary

* `patient_layout_coords.csv`: 2D coordinates exported from the network layout
* **2D plot**: produced by `Network_graph_ploting.py` (after updating `a, b`)
* **Evolution HTML**: produced by `Network_Evolution_html_generation.py`
* **Evolution GIF**: produced by `convert_html_2_gif.py`

---
