import streamlit as st # type: ignore
import numpy as np # type: ignore
import plotly.graph_objs as go # type: ignore

# Function to run the Gierer-Meinhardt simulation based on selected model
def without_diffusion_gm_system1(system, T, A_0, H_0, delta_t, t_end, k=0.01, c=0.01):
    epsilon = 1e-10  # Small value to avoid division by zero

    # Number of time steps
    N = int(t_end / delta_t)
    
    # Initialise arrays to store the values
    A = np.zeros(N)
    H = np.zeros(N)
    t = np.linspace(0, t_end, N)  # Array to store time points

    # Set initial conditions
    A[0] = A_0
    H[0] = H_0

    # Choose the simulation based on the selected system
    if system == "GM 1 - Without diffusion":
        for n in range(N-1):
            A[n+1] = A[n] + delta_t * (-A[n] + (A[n]**2) / (H[n] + epsilon))
            H[n+1] = H[n] + delta_t * ((-mu*H[n] + A[n]**2) / T)
    elif system == "GM 2 - Without diffusion, with activator saturation":
        for n in range(N-1):
            A[n+1] = A[n] + delta_t * (-A[n] + (A[n]**2) / ((H[n] + epsilon) * (1 + k * A[n]**2)))
            H[n+1] = H[n] + delta_t * ((-mu*H[n] + A[n]**2) / T)
    elif system == "GM 3 - Without diffusion, with basic activator production":
        for n in range(N-1):
            A[n+1] = A[n] + delta_t * (-A[n] + ((A[n]**2) / (H[n] + epsilon)) + c)
            H[n+1] = H[n] + delta_t * ((-mu*H[n] + A[n]**2) / T)

    return A, H, t


# Streamlit app layout
st.title("GM System Without Diffusion")

# Sidebar for user inputs
st.sidebar.header("Parameters")

# Dropdown for selecting the GM system
system = st.sidebar.selectbox(
    "Select GM System",
    options=[
        "GM 1 - Without diffusion",
        "GM 2 - Without diffusion, with activator saturation",
        "GM 3 - Without diffusion, with basic activator production"
    ]
)

# Parameters input
T = st.sidebar.number_input("Reaction time constant (τ)", value=1.11, min_value=0.01, step=0.01)
A_0 = st.sidebar.number_input("Initial condition for A", value=0.9, min_value=0.0, step=0.01)
H_0 = st.sidebar.number_input("Initial condition for H", value=0.9, min_value=0.0, step=0.01)
delta_t = st.sidebar.number_input("Time step (Δt)", value=0.001, min_value=0.0001, step=0.0001)
t_end = st.sidebar.number_input("End time", value=100.0, min_value=0.01, step=0.1)
mu = st.sidebar.number_input("Inhibitor decay rate (μ)", value=1, min_value=1, step=1)

# Only show k and c parameters if they apply to the selected system
if system == "GM 2 - Without diffusion, with activator saturation":
    k = st.sidebar.number_input("Parameter k", value=0.01, min_value=0.01, step=0.01)
else:
    k = None

if system == "GM 3 - Without diffusion, with basic activator production":
    c = st.sidebar.number_input("Parameter c", value=0.01, min_value=0.01, step=0.01)
else:
    c = None

# Run the simulation based on the selected system
A, H, t = without_diffusion_gm_system1(system, T, A_0, H_0, delta_t, t_end, k, c)

# Plot H vs A
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=A, y=H, mode='lines', name='A vs H', line=dict(color='green')))
fig1.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='(0, 0)', marker=dict(color='blue')))
fig1.add_trace(go.Scatter(x=[1], y=[1], mode='markers', name='(1, 1)', marker=dict(color='red')))
fig1.add_trace(go.Scatter(x=[A[0]], y=[H[0]], mode='markers', name= f'Initial Point - ({A[0]:.2f}, {H[0]:.2f})', marker=dict(color='yellow')))
fig1.add_trace(go.Scatter(x=[A[-1]], y=[H[-1]], mode='markers', name=f'End Point - ({A[-1]:.2f}, {H[-1]:.2f})', marker=dict(color='purple')))

fig1.update_layout(
    title="Activator (A) vs Inhibitor (H)",
    xaxis_title="Activator (A) Concentration",
    yaxis_title="Inhibitor (H) Concentration",
    showlegend=True
)

# Show H vs A plot in Streamlit
st.plotly_chart(fig1)

# Plot A and H vs t as separate time series
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=A, mode='lines', name='Activator A', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=t, y=H, mode='lines', name='Inhibitor H', line=dict(color='orange')))
fig2.update_layout(
    title="Concentration of A and H over Time",
    xaxis_title="Time (t)",
    yaxis_title="Concentration",
    showlegend=True
)

# Show A and H vs t plot in Streamlit
st.plotly_chart(fig2)