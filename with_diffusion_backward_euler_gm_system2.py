import streamlit as st  # type: ignore

st.set_page_config(page_title="Biological Pattern Formation - GM Model", layout="wide")

# Title
st.title("🧬 Biological Pattern Formation for the Gierer–Meinhardt System")

# Overview
st.markdown("""
This interactive web application focusses on simulating and exploring **biological pattern formation** through the **Gierer–Meinhardt model**, a classic framework for modelling **activator–inhibitor dynamics** in developmental biology.

---

### 🎯 Aim of the Project

The primary aim of this project is to:

> **Investigate the dynamics and pattern-forming capabilities of the Gierer–Meinhardt system**, both with and without spatial diffusion, by examining multiple model variants. Through numerical simulations, we aim to identify conditions under which the system exhibits steady states, oscillations, or spatial pattern formation — with a focus on the emergence of Turing instability.

---

### 📘 Background

The **Gierer–Meinhardt model** describes interactions between:

- **Activator (A)** – promotes its own production and that of the inhibitor.
- **Inhibitor (H)** – suppresses the activator.

These interactions, coupled with spatial diffusion, can lead to the emergence of **spatial patterns**, such as those observed in animal skin or plant structures.

---

### 📐 Gierer–Meinhardt Model Variants

#### GM 1 – Basic System
$$
\\frac{\\partial A}{\\partial t} = D_A \\nabla^2 A - A + \\frac{A^2}{H}
$$
$$
\\tau \\frac{\\partial H}{\\partial t} = D_H \\nabla^2 H - \\mu H + A^2
$$

#### GM 2 – With Activator Saturation
$$
\\frac{\\partial A}{\\partial t} = D_A \\nabla^2 A - A + \\frac{A^2}{H(1 + k A^2)}
$$
$$
\\tau \\frac{\\partial H}{\\partial t} = D_H \\nabla^2 H - \\mu H + A^2
$$

#### GM 3 – With Constant Activator Production
$$
\\frac{\\partial A}{\\partial t} = D_A \\nabla^2 A - A + \\frac{A^2}{H} + c
$$
$$
\\tau \\frac{\\partial H}{\\partial t} = D_H \\nabla^2 H - \\mu H + A^2
$$

> For simulations **without diffusion**, the spatial Laplacian terms ($$ D_A \\nabla^2 A $$ and $$ D_H \\nabla^2 H $$) are omitted.

---

### 🔍 Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| **$$ A, H $$** | Activator and inhibitor concentrations |
| **$$ D_A, D_H $$** | Diffusion coefficients for activator and inhibitor |
| **$$ \\tau $$** | Inhibitor response time |
| **$$ \\mu $$** | Inhibitor decay rate |
| **$$ k $$** | Saturation constant (GM 2 only) |
| **$$ c $$** | Constant activator production (GM 3 only) |

---

### 🧭 Navigation

Use the sidebar to:
- Simulate each GM system numerically.
- Explore convergence, oscillatory behaviour, and stability.
- Add spatial diffusion and visualise Turing patterns.

---
""")