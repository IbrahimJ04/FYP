import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Laplacian operator for finite differences
def laplacian_operator(U, V, dx):
    Lu = (U[0:-2, 1:-1] + U[1:-1, 0:-2] + U[1:-1, 2:] + U[2:, 1:-1] - 4*U[1:-1, 1:-1]) / dx**2
    Lv = (V[0:-2, 1:-1] + V[1:-1, 0:-2] + V[1:-1, 2:] + V[2:, 1:-1] - 4*V[1:-1, 1:-1]) / dx**2
    return Lu, Lv

# Gierer-Meinhardt model simulation with smooth transitions in a single grid
def GM(params, initial_matrices, timesteps, frameMod, transition_time):
    Du, Dv, rho, kappa, mu, ku, kv, sv, dt, dx = (params['Du'], params['Dv'], params['rho'], params['kappa'],
                                                  params['mu'], params['ku'], params['kv'], params['sv'],
                                                  params['dt'], params['dx'])
    U, V = initial_matrices
    u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]

    # Set up the figure (a single plot that updates)
    fig, ax = plt.subplots()
    ax.axis('off')  # Remove axes for a clean plot
    im = ax.imshow(v, cmap=params["myCmap"], extent=[-1, 1, -1, 1])
    placeholder = st.empty()  # Create a placeholder for the plot in Streamlit

    # Simulation loop
    for i in range(timesteps):
        Lu, Lv = laplacian_operator(U, V, dx)
        vv = v * v
        sv = rho * (vv / (u * (1 + kappa * vv)) - mu * v) + Dv * Lv
        su = rho * (vv - ku * u) + Du * Lu
        u += dt * su
        v += dt * sv

        if i % frameMod == 0:
            # Update the data for the same plot, no new plot created
            im.set_data(v)
            placeholder.pyplot(fig)  # Replace the previous plot in Streamlit

            time.sleep(transition_time)  # Pause for a short time to create smooth transitions

# Main function to run the Streamlit app
def main():
    st.title("Gierer-Meinhardt Reaction-Diffusion Simulation")

    # Simulation parameters for the Gierer-Meinhardt (GM) model
    params = {
        "Du": 2.0, "Dv": 0.1, "rho": 0.5, "kappa": 0.238, "mu": 1.0,
        "ku": 0.9, "kv": 1.0, "sv": 0.3, "dt": 0.1, "dx": 1.0, 
        "myCmap": plt.cm.copper
    }

    # Sidebar inputs for timesteps and grid size
    timesteps = st.sidebar.slider("Timesteps", min_value=1000, max_value=10000, value=5000, step=1000)
    size = st.sidebar.slider("Grid Size", min_value=50, max_value=300, value=200, step=10)
    transition_time = st.sidebar.slider("Transition Time (seconds)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

    # Grid initialization (random seed)
    U = np.ones((size, size))
    V = np.zeros((size, size))
    V += 0.25 * (np.random.rand(size, size) - 0.5)  # Random noise in V

    # Initial matrices
    initial_matrices = (U, V)

    if st.button("Run Simulation"):
        frameMod = timesteps // 250  # Set frame modification rate
        GM(params, initial_matrices, timesteps, frameMod, transition_time)

if __name__ == "__main__":
    main()
