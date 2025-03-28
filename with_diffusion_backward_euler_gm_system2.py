import streamlit as st  # type: ignore

import matplotlib.pyplot as plt

st.set_page_config(page_title="Biological Pattern Formation - GM Model", layout="wide")

# Title
st.title("🧬 Biological Pattern Formation")
st.subheader("Gierer-Meinhardt Model Simulation and Analysis")

# Description
st.markdown("""
Welcome to this interactive web app that explores the fascinating world of **biological pattern formation** using the **Gierer-Meinhardt (GM) model**.

This project investigates the mathematical modeling of **activator-inhibitor systems** in biology, focusing on how stable patterns like spots and stripes emerge. The Gierer-Meinhardt model consists of nonlinear reaction terms with and without diffusion, simulating the interactions between:

- **Activator (A)**: Promotes its own production and that of the inhibitor.
- **Inhibitor (H)**: Suppresses activator production and diffuses over a longer range.

The system is known for exhibiting **Turing instability**, which underlies pattern formation in morphogenesis.

---

### 📐 Key Model Equations (Without Diffusion)

""")

# GM1
st.markdown("#### GM 1 – Basic System")
st.latex(r"""
A_{n+1} = A_n + \Delta t \left(-A_n + \frac{A_n^2}{H_n + \epsilon}\right)
""")
st.latex(r"""
H_{n+1} = H_n + \Delta t \left(\frac{-H_n + A_n^2}{\tau} \right)
""")

# GM2
st.markdown("#### GM 2 – With Activator Saturation")
st.latex(r"""
A_{n+1} = A_n + \Delta t \left(-A_n + \frac{A_n^2}{(H_n + \epsilon)(1 + k A_n^2)}\right)
""")
st.latex(r"""
H_{n+1} = H_n + \Delta t \left(\frac{-H_n + A_n^2}{\tau} \right)
""")

# GM3
st.markdown("#### GM 3 – With Constant Activator Production")
st.latex(r"""
A_{n+1} = A_n + \Delta t \left(-A_n + \frac{A_n^2}{H_n + \epsilon} + c\right)
""")
st.latex(r"""
H_{n+1} = H_n + \Delta t \left(\frac{-H_n + A_n^2}{\tau} \right)
""")

# Parameters Explanation
st.markdown("""
---

### 🔍 Parameter Descriptions

- **A, H**: Concentrations of activator and inhibitor.
- **Δt**: Time step used in the numerical simulation.
- **τ (tau)**: Inhibitor response time.
- **ε**: A small constant to avoid division by zero.
- **k**: Saturation constant controlling nonlinear activator feedback.
- **c**: Constant activator production rate.

---

This simulation is implemented using the **Forward Euler method**, a simple numerical integration technique suitable for exploring oscillations, convergence, and stability behavior in dynamic systems.

Use the navigation sidebar to explore simulations of each GM variant, examine oscillations, convergence, and eventually spatial pattern formation with diffusion!
""")

# Footer
st.markdown("---")
st.info("Developed as part of a research project on biological pattern formation. Explore more via the sidebar!")

