import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, shapiro, probplot, norm  
import pingouin as pg  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configure Streamlit
st.title("Surgery Reaction Time Analysis")
st.markdown("""
This app analyzes patient reaction times during two types of surgeries: VR and CD.  
Upload your dataset and explore the statistical results and visualizations.
""")

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Shapiro-Wilk Test", "Statistical Tests"])

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Function to save plots as images
def save_plot_as_image(fig, filename):
    local_directory = r'C:\StatsProjects\plots'  # Define custom directory
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    file_path = os.path.join(local_directory, f'{filename}.png')
    fig.savefig(file_path, bbox_inches='tight')
    return file_path

# If a file is uploaded, perform analysis
if uploaded_file:
    # Load the dataset
    df_wide = pd.read_csv(uploaded_file)

    # Remove unnecessary columns
    df_wide = df_wide.drop(columns=[col for col in df_wide.columns if "Unnamed" in col])

    # Convert from wide to long format
    df_long = df_wide.melt(id_vars=['Participants'], var_name='SurgeryTime', value_name='ReactionTime')

    # Extract 'Surgery' and 'Time' from the 'SurgeryTime' column
    df_long['Surgery'] = df_long['SurgeryTime'].str.extract(r'([A-Za-z]+)')  # Surgery type (VR/CD)
    df_long['Time'] = df_long['SurgeryTime'].str.extract(r'(\d+)').astype(int)  # Time (10, 30, 50)

    # Drop the original 'SurgeryTime' column
    df_long = df_long.drop(columns=['SurgeryTime'])

    # Function to perform Shapiro-Wilk test and generate plots
    def shapiro_test(data, label):
        data = data.to_numpy()
        stat, p_value = shapiro(data)
        st.write(f"Shapiro-Wilk Test for {label}: Statistic = {stat:.3f}, p-value = {p_value:.3f}")
        if p_value > 0.05:
            st.write(f"{label} is normally distributed (Fail to reject H0).")
        else:
            st.write(f"{label} is not normally distributed (Reject H0).")

        # Generate QQ plot
        fig, ax = plt.subplots()
        probplot(data, dist="norm", plot=ax)
        st.subheader(f"QQ Plot for {label}")
        st.pyplot(fig)

        # Generate Histogram + Normal Curve
        fig2, ax2 = plt.subplots()
        sns.histplot(data, kde=True, stat='density', color='skyblue', ax=ax2)
        xmin, xmax = ax2.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(data), np.std(data))
        ax2.plot(x, p, 'k', linewidth=2)
        ax2.set_title(f'Normal Distribution Plot for {label}')
        st.pyplot(fig2)

        # Save plots
        save_plot_as_image(fig, f'qq_plot_{label}')
        save_plot_as_image(fig2, f'normal_dist_plot_{label}')

    # Perform Friedman Test for VR Surgery
    def perform_friedman(data):
        grouped = data.pivot(index='Participants', columns='Time', values='ReactionTime')
        stat, p_value = friedmanchisquare(*[grouped[col] for col in grouped.columns])
        st.write(f"Friedman Test for VR Surgery: Statistic = {stat:.2f}, p-value = {p_value:.2e}")
        return p_value

    # Perform Wilcoxon Signed-Rank Test for pairwise comparisons (post-hoc)
    def pairwise_wilcoxon(data):
        # Perform pairwise comparisons between 10, 30, 50 seconds using Wilcoxon Signed-Rank Test
        results = []
        times = [10, 30, 50]
        for i in range(len(times)):
            for j in range(i+1, len(times)):
                time_1 = times[i]
                time_2 = times[j]
                group1 = data[data['Time'] == time_1]['ReactionTime']
                group2 = data[data['Time'] == time_2]['ReactionTime']
                stat, p_value = wilcoxon(group1, group2)
                results.append((f"Time {time_1} vs Time {time_2}", stat, p_value))
        
        # Create a DataFrame to display results
        pairwise_df = pd.DataFrame(results, columns=['Comparison', 'Statistic', 'p-value'])
        st.write("### Pairwise Comparisons (Wilcoxon Signed-Rank Test) between Time Points:")
        st.write(pairwise_df)

    # Perform Repeated Measures ANOVA for CD Surgery
    def perform_rm_anova(data):
        anova_result = pg.rm_anova(dv='ReactionTime', within='Time', subject='Participants', data=data, detailed=True)
        st.write("Repeated Measures ANOVA for CD Surgery:")
        st.write(anova_result)
        return anova_result

    # Perform pairwise t-tests for CD Surgery with Bonferroni correction
    def pairwise_t_tests(data):
        times = [10, 30, 50]
        results = []
        for i in range(len(times)):
            for j in range(i+1, len(times)):
                time_1 = times[i]
                time_2 = times[j]
                group1 = data[data['Time'] == time_1]['ReactionTime']
                group2 = data[data['Time'] == time_2]['ReactionTime']
                stat, p_value = ttest_rel(group1, group2)
                # Apply Bonferroni correction
                corrected_p_value = min(p_value * 3, 1.0)
                results.append((f"Time {time_1} vs Time {time_2}", stat, p_value, corrected_p_value))
        
        # Create a DataFrame to display results
        pairwise_df = pd.DataFrame(results, columns=['Comparison', 'Statistic', 'p-value', 'Corrected p-value'])
        st.write("### Pairwise Comparisons (t-tests with Bonferroni Correction) between Time Points:")
        st.write(pairwise_df)

    # Separate data by surgery
    vr_data = df_long[df_long['Surgery'] == 'VR']
    cd_data = df_long[df_long['Surgery'] == 'CD']

    if page == "Shapiro-Wilk Test":
        st.subheader("Normality Test (Shapiro-Wilk Test)")
        for surgery in ['VR', 'CD']:
            for time_point in [10, 30, 50]:
                data = df_long[(df_long['Surgery'] == surgery) & (df_long['Time'] == time_point)]['ReactionTime']
                label = f"{surgery} Surgery at {time_point} seconds"
                shapiro_test(data, label)

    elif page == "Statistical Tests":
        st.subheader("Statistical Test Results")

        # VR Surgery: Friedman Test
        st.write("### VR Surgery (Friedman Test)")
        perform_friedman(vr_data)

        # Post-hoc pairwise comparisons (Wilcoxon)
        pairwise_wilcoxon(vr_data)

        # CD Surgery: Repeated Measures ANOVA
        st.write("### CD Surgery (Repeated Measures ANOVA)")
        perform_rm_anova(cd_data)

        # Post-hoc pairwise comparisons (t-tests with Bonferroni)
        pairwise_t_tests(cd_data)

else:
    st.info("Please upload a CSV file to begin.")
