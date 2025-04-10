import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import plotly.graph_objs as go  # type: ignore
import scipy.signal
# For solving linear systems
import scipy.linalg  # type: ignore


# Define GM System with np.roll for periodic boundary conditions
def ftcs_gm_system_with_diffusion(system, T, A_0, H_0, delta_t, delta_x, t_end, x_end, D_A, D_H, k=None, c=None):
    epsilon = 1e-10  # Small value to avoid division by zero

    N_t = int(t_end / delta_t) + 1  # Number of time steps
    N_x = int(x_end / delta_x)      # Number of spatial points

    # Initialise arrays
    A = np.zeros((N_x, N_t))
    H = np.zeros((N_x, N_t))

    # Set initial conditions
    A[:, 0] = A_0
    H[:, 0] = H_0

    # Check for stability condition
    max_D = max(D_A, D_H)
    if delta_t > delta_x**2 / (2 * max_D):
        st.warning(f"Time step (delta_t) is too large for stability. Reduce it to less than {delta_x**2 / (2 * max_D):.5f}")

    # Forward-Time Central-Space (FTCS) method
    for n in range(N_t - 1):  # Iterate over time --> 'for n in range(3) --> gives (0, 1, 2)'
        
        # Apply periodic boundary conditions using 'np.roll'
        A_left = np.roll(A[:, n], 1)   # Left neighbour
        A_right = np.roll(A[:, n], -1) # Right neighbour
        H_left = np.roll(H[:, n], 1)   # Left neighbour
        H_right = np.roll(H[:, n], -1) # Right neighbour

        # Laplacians calculated using Central Finite Difference Approximation
        A_xx = (A_right - 2 * A[:, n] + A_left) / delta_x**2
        H_xx = (H_right - 2 * H[:, n] + H_left) / delta_x**2


        if system == "GM 1 - With diffusion":
            A[:, n + 1] = A[:, n] + delta_t * (D_A * A_xx - A[:, n] + (A[:, n]**2) / (H[:, n] + epsilon))
            H[:, n + 1] = H[:, n] + delta_t * (D_H * H_xx - mu*H[:, n] + A[:, n]**2 / T)
        elif system == "GM 2 - With diffusion, with activator saturation" and k is not None:
            A[:, n + 1] = A[:, n] + delta_t * (
                D_A * A_xx - A[:, n] + ((A[:, n]**2) / ((H[:, n] + epsilon) * (1 + k * A[:, n]**2)))
            )
            H[:, n + 1] = H[:, n] + delta_t * (D_H * H_xx - mu*H[:, n] + A[:, n]**2 / T)
        elif system == "GM 3 - With diffusion, with basic activator production" and c is not None:
            A[:, n + 1] = A[:, n] + delta_t * (D_A * A_xx - A[:, n] + (A[:, n]**2) / (H[:, n] + epsilon) + c)
            H[:, n + 1] = H[:, n] + delta_t * (D_H * H_xx - mu*H[:, n] + A[:, n]**2 / T)

        # Check for invalid values
        if np.any(np.isnan(A[:, n + 1])) or np.any(np.isnan(H[:, n + 1])):
            st.error("Simulation encountered invalid values. Adjust parameters.")
            break

    return A, H, np.linspace(0, x_end, N_x), np.linspace(0, t_end, N_t)


# Function to count spatial peaks in final activator concentration
def count_spatial_peaks(A_final, x_vals):
    peaks, _ = scipy.signal.find_peaks(A_final, height=0.5)  # Adjust height threshold if needed
    
    return len(peaks), x_vals[peaks], A_final[peaks]  # Return peak count and peak positions


# D_N_star equation from paper titled: 'The Stability of Spike Solutions to the One-Dimensional' --> Eqn. (4.65)
def expected_peak_count(D_H, mu, N_max=100):
    epsilon = 1e-10
    for N in range(1, N_max + 1):
        cos_term = np.cos(np.pi / N)
        theta_N = (N / 2) * np.log(2 + cos_term + np.sqrt((2 + cos_term) ** 2 - 1))
        D_N_star = mu / ((theta_N ** 2) + epsilon)

        if D_H >= D_N_star:
            return max(1, N - 1)  # Previous N was the last valid one

    return N_max  # If D_H < D_N_star even at max N




## Streamlit app layout
st.title("GM System with Diffusion")

# Sidebar for user inputs
st.sidebar.header("Parameters")

# Dropdown for selecting the GM system
system = st.sidebar.selectbox(
    "Select GM System",
    options=[
        "GM 1 - With diffusion",
        "GM 2 - With diffusion, with activator saturation",
        "GM 3 - With diffusion, with basic activator production"
    ]
)

# Parameter inputs
T = st.sidebar.number_input("Reaction time constant (τ)", value=1.11, min_value=0.01, step=0.01)
delta_t = st.sidebar.number_input("Time step (Δt)", value=0.004, min_value=0.0001, step=0.001, format="%.3f")
delta_x = st.sidebar.number_input("Space step (Δx)", value=0.1, min_value=0.01, step=0.01)
t_end = st.sidebar.number_input("End time", value=25.0, min_value=0.01, step=0.1)
x_end = st.sidebar.number_input("End space", value=10.0, min_value=0.01, step=0.1)
D_A = st.sidebar.number_input("Diffusion coefficient for A", value=0.01, min_value=0.0, step=0.001)
D_H = st.sidebar.number_input("Diffusion coefficient for H", value=1.0, min_value=0.0, step=0.001)
mu = st.sidebar.number_input("Inhibitor decay rate (μ)", value=1.0, min_value=0.01, step=0.01)
r = st.sidebar.number_input("Integer r for initial condition", value=1, min_value=1, step=1)

# Generate spatial variable x
x_vals = np.linspace(0, x_end, int(x_end / delta_x))

# Define initial conditions based on sinusoidal perturbation
A_0 = mu + 0.1 * np.sin(2 * np.pi * r * x_vals)
H_0 = mu + 0.1 * np.sin(2 * np.pi * r * x_vals)

# Conditionally display k and c parameters
k = st.sidebar.number_input("Parameter k", value=0.01, min_value=0.01, step=0.01) if system == "GM 2 - With diffusion, with activator saturation" else None
c = st.sidebar.number_input("Parameter c", value=0.01, min_value=0.01, step=0.01) if system == "GM 3 - With diffusion, with basic activator production" else None

# Run the simulation
A, H, x_vals, t_vals = ftcs_gm_system_with_diffusion(system, T, A_0, H_0, delta_t, delta_x, t_end, x_end, D_A, D_H, k, c)




### PLOTS ###

# 1. 3D Surface Plot for Activator A over Space and Time
st.markdown("""
---
""")

fig1 = go.Figure(data=[go.Surface(z=A, x=t_vals, y=x_vals, colorscale="Earth")])
fig1.update_layout(title="3D Surface Plot of Activator (A)", scene=dict(
    xaxis_title="Time",
    yaxis_title="Space",
    zaxis_title="Activator A Concentration"
))
st.plotly_chart(fig1)



# 2. 3D Surface Plot for Inhibitor H over Space and Time
st.markdown("""
---
""")

fig2 = go.Figure(data=[go.Surface(z=H, x=t_vals, y=x_vals, colorscale="Earth")])
fig2.update_layout(title="3D Surface Plot of Inhibitor (H)", scene=dict(
    xaxis_title="Time",
    yaxis_title="Space",
    zaxis_title="Inhibitor H Concentration"
))
st.plotly_chart(fig2)



# 3. Line Plot Over Time at a Fixed Spatial Location
st.markdown("""
---
""")

fixed_location = st.slider(
    "Select Spatial Location (Index):", 
    min_value=0, 
    max_value=len(x_vals) - 1, 
    value=len(x_vals) // 2,  # Default is the midpoint
    step=1
)

# Display the corresponding spatial point
selected_spatial_point = x_vals[fixed_location]

# Line Plot Over Time at the Selected Spatial Location
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t_vals, y=A[fixed_location, :], mode='lines', name='Activator A', line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=t_vals, y=H[fixed_location, :], mode='lines', name='Inhibitor H', line=dict(color='orange')))
fig3.update_layout(
    title=f"Concentration Over Time at Spatial Location: {selected_spatial_point:.2f}",
    xaxis_title="Time",
    yaxis_title="Concentration",
    margin=dict(t=50, b=50),
)
st.plotly_chart(fig3)



# 4. Count peaks in final activator concentration
st.markdown("""
---
""")

# Compute peaks and compare with theory
A_final = A[:, -1]  # Extract final activator concentration
H_final = H[:, -1]  # Extract final inhibitor concentration
num_peaks, peak_x, peak_A = count_spatial_peaks(A_final, x_vals)


# Compute the critical diffusion coefficient for the detected number of peaks
d = D_H / D_A
d_min = mu * (3 + 2 * np.sqrt(2))  # ≈ 5.83μ
D_H_turing = D_A * d_min    # Global threshold for Turing instability

st.write("#### Theoretical Turing Instability Check")
st.write(f"###### • Current D_H (Simulation): {D_H:.4f}")
st.write(f"###### • Global Turing Threshold (D_H > D_A · 5.83μ): {D_H_turing:.4f}")


if D_H > D_H_turing:
    st.success("Turing instability condition is satisfied: pattern formation is theoretically expected.")
else:
    st.info("Turing instability condition not satisfied: the steady state is expected to remain stable (no pattern formation). If there appears to be any patterns, these are merely the initial perturbations which will eventually die out over time.")

# Simulation-based peak detection results
if num_peaks == 0:
    st.warning("No spatial peaks were detected in the final output.")

    if D_H > 10 * D_H_turing: # i.e. D_H is a lot larger than the critical value
        st.info("Note: For very large D_H the solution should always show one peak (according to the shadow problem), however this may not be seen due to numerical complications for large D_H. Try decreasing the spatial step size (Δx).")


# Plot activator and inhibitor concentrations across space at final time step
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=x_vals, y=A_final, mode="lines", name="Activator A", line=dict(color='blue')))
fig4.add_trace(go.Scatter(x=x_vals, y=H_final, mode="lines", name="Inhibitor H", line=dict(color='orange')))
fig4.add_trace(go.Scatter(x=peak_x, y=peak_A, mode="markers", name="Detected Peaks", marker=dict(color='red', size=8)))

fig4.update_layout(
    title="Final Activator and Inhibitor Concentrations with Detected Peaks",
    xaxis_title="Space",
    yaxis_title="Concentration",
)
st.plotly_chart(fig4)


expected_N = expected_peak_count(D_H, mu)
st.write(f"###### • Number of Spatial Peaks in Activator A (Simulation): {num_peaks}")
st.write(f"###### • Theoretical number of stable peaks expected (from equation below): {expected_N}")


st.markdown("**Estimating expected number of peaks** $N$ **given** $D_H$:")
st.latex(r"""
\text{Find the largest } N \text{ such that } D_H < D_N^* = \frac{\mu}{\theta_N^2}
""")
st.latex(r"""
\theta_N = \frac{N}{2} \cdot \ln\left( 2 + \cos\left( \frac{\pi}{N} \right) + \sqrt{\left( 2 + \cos\left( \frac{\pi}{N} \right) \right)^2 - 1} \right)
""")



# 5. Animation Over Time Showing Concentration Across Space
st.markdown("""
---
""")

frame_duration = 0.1 # Adjust animation speed

# Animation Over Time Showing Concentration Across Space
frames = [
    go.Frame(
        data=[
            # Activator A line plot
            go.Scatter(x=x_vals, y=A[:, i], mode="lines", name="Activator A", line=dict(color='blue')),
            
            # Inhibitor H line plot
            go.Scatter(x=x_vals, y=H[:, i], mode="lines", name="Inhibitor H", line=dict(color='orange')),

            # Marker for selected spatial point for Activator A
            go.Scatter(
                x=[selected_spatial_point], y=[A[fixed_location, i]], mode="markers",
                marker=dict(size=7, color="red", symbol="circle", line=dict(color="black", width=1)),
                name=f"Spatial Point = {selected_spatial_point:.2f} (Activator A) at t = {t_vals[i]:.2f}"
            ),

            # Marker for selected spatial point for Inhibitor H
            go.Scatter(
                x=[selected_spatial_point], y=[H[fixed_location, i]], mode="markers",
                marker=dict(size=7, color="rgb(0, 204, 204)", symbol="circle", line=dict(color="black", width=1)),
                name=f"Spatial Point = {selected_spatial_point:.2f} (Inhibitor H) at t = {t_vals[i]:.2f}"
            )
        ],
        name=f"t = {t_vals[i]:.2f}",
        layout=go.Layout(
            annotations=[
                dict(
                    x=0.5, y=1.1, xref="paper", yref="paper", showarrow=False,
                    text=f"Simulation Time: {t_vals[i]:.2f} units", font=dict(size=16, color="black")
                )
            ]
        )
    )
    for i in range(len(t_vals))
]

fig5 = go.Figure(
    data=[
        # Activator A and Inhibitor H lines at t=0
        go.Scatter(x=x_vals, y=A[:, 0], mode="lines", name="Activator A", line=dict(color='blue')),
        go.Scatter(x=x_vals, y=H[:, 0], mode="lines", name="Inhibitor H", line=dict(color='orange')),

        # Markers for selected spatial points (at t=0) for both A and H
        go.Scatter(
            x=[selected_spatial_point], y=[A[fixed_location, 0]], mode="markers",
            marker=dict(size=7, color="red", symbol="circle", line=dict(color="black", width=1)),
            name=f"Spatial Point = {selected_spatial_point:.2f} (Activator A) at t = 0"
        ),
        go.Scatter(
            x=[selected_spatial_point], y=[H[fixed_location, 0]], mode="markers",
            marker=dict(size=7, color="rgb(0, 204, 204)", symbol="circle", line=dict(color="black", width=1)),
            name=f"Spatial Point = {selected_spatial_point:.2f} (Inhibitor H) at t = 0"
        )
    ],
    layout=go.Layout(
        title="Concentration Across Space Over Time",
        xaxis=dict(title="Space"),
        yaxis=dict(title="Concentration"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None, 
                        {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None], 
                        {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}
                    ]
                )
            ]
        )],
        annotations=[
            dict(
                x=0.5, y=1.1, xref="paper", yref="paper", showarrow=False,
                text="Simulation Time: 0.00 units", font=dict(size=16, color="black")
            )
        ]
    ),
    frames=frames
)
st.plotly_chart(fig5)


