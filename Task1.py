import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
import plotly.express as px
from geopy.geocoders import Nominatim
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as ssd
import statsmodels.stats.outliers_influence as oi

# --------------------------------------------
# Constants & Configuration
# --------------------------------------------
DATA_PATH = r"E:\ERU\Level 3\S1\Data Analysis\Project\happy_clean.csv"  # Define the location of data

# Set Streamlit app title and icon
st.set_page_config(
    page_title="Happiness Score Analysis :earth_africa:",
    page_icon=":smiley:",  # You can use any emoji
    layout="wide",  # Use the full width of the screen
)

# Add some visual effects
st.balloons()

# --------------------------------------------
# Load and Cache Data
# --------------------------------------------
@st.cache_data
def load_data(path):
    """Loads the data from the given path and caches it."""
    return pd.read_csv(path)

data = load_data(DATA_PATH)

# --------------------------------------------
# Data Preparation Functions
# --------------------------------------------
@st.cache_data
def get_iso_code(country_name):
    """
    Retrieves the ISO 3166-1 alpha-3 code for a given country name.

    Args:
        country_name (str): The name of the country.

    Returns:
        str: The ISO code if found, otherwise None.
    """
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_3
    except LookupError:
        return None

# Add ISO Code Column
data["ISO_Code"] = data["Country"].apply(get_iso_code)


# --------------------------------------------
# Streamlit App Layout
# --------------------------------------------

# Main title and Introduction
st.title("Happiness Score Analysis :smiley:")
st.markdown("Explore the factors influencing global happiness scores through interactive visualizations and statistical analysis.")

# Create Tabs
analysis_tab, eda_tab, models_tab = st.tabs(["Analysis", "EDA", "Models"])


# --------------------------------------------
# Analysis Tab
# --------------------------------------------

with analysis_tab:
    st.header("Data Overview and Global Happiness :earth_africa:", anchor="analysis")

    # Data Overview
    st.subheader("Raw Data")
    st.dataframe(data.head(10))

    st.subheader("Data Shape")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")

    st.subheader("Data Information")
    st.write(data.info())

    # Happiness Ranking
    st.header("Happiness Ranking :trophy:")

    st.subheader("Top 5 Happiest Countries")
    top_5 = data.nlargest(5, "Happiness_score")
    fig_top = px.bar(top_5, x="Country", y="Happiness_score", title="Top 5 Countries by Happiness Score")
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Lowest 5 Happiest Countries")
    bottom_5 = data.nsmallest(5, "Happiness_score")
    fig_bottom = px.bar(bottom_5, x="Country", y="Happiness_score", title="Lowest 5 Countries by Happiness Score")
    st.plotly_chart(fig_bottom, use_container_width=True)

    # Continental Analysis
    st.header("Continental Analysis :globe_with_meridians:")

    st.subheader("Average Happiness Score by Continent")
    continent_scores = data.groupby("continent")["Happiness_score"].mean().sort_values(ascending=False)
    st.dataframe(continent_scores)

    st.subheader("Happiness Score by Continent (Choropleth Map)")
    continent_mapping = {
        "World": None,
        "Asia": "asia",
        "Africa": "africa",
        "Europe": "europe",
        "North America": "north america",
        "South America": "south america",
        "Oceania": "oceania",
    }

    selected_continent = st.selectbox("Select Continent for Map", list(continent_mapping.keys()))

    fig_map = px.choropleth(
        data, locations="ISO_Code", color="Happiness_score",
        hover_name="Country", color_continuous_scale="Agsunset",
        projection="natural earth", title=f'Happiness Score in {selected_continent}'
    )

    if selected_continent != "World":
        fig_map.update_geos(
            visible=False, resolution=110, scope=continent_mapping[selected_continent],
            showcountries=True, countrycolor="Black",
            showsubunits=True, subunitcolor="Blue"
        )

    fig_map.update_layout(height=600, width=800)
    st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------
# EDA Tab
# --------------------------------------------

with eda_tab:
    st.header("Exploratory Data Analysis :bar_chart:", anchor = "eda")

    # Feature Distribution
    st.subheader("Feature Distribution")

    st.subheader("Happiness Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    data["Happiness_score"].hist(ax=ax, bins=20, grid=False)
    ax.set_xlabel("Happiness Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Boxplots for Numerical Features")
    num_data = data.select_dtypes(include="number")

    num_cols = st.slider("Number of Columns for Boxplots:", min_value=1, max_value=4, value=2)
    num_rows = (len(num_data.columns) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(num_data.columns):
        sns.boxplot(y=num_data[column], ax=axes[i])
        axes[i].set_title(column)

    for i in range(len(num_data.columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Analysis
    st.header("Correlation Analysis :chart_with_upwards_trend:")

    st.subheader("Scatter Matrix of Features")
    fig = plt.figure(figsize=(15, 15))
    pd.plotting.scatter_matrix(data.select_dtypes(include=np.number), figsize=(15, 15), diagonal="kde")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True, ax=ax)
    plt.title("Correlation Heatmap of Numerical Features")
    st.pyplot(fig)

    # Outlier Analysis
    st.header("Outlier Analysis and Model Refinement :warning:")

    st.subheader("Influence Analysis Plots")
    try:
        lm_trans_y = smf.ols(
            "np.log(Happiness_score) ~ GDP_per_capita + Social_support + Healthy_life_expectancy",
            data=data).fit()
        influence = oi.OLSInfluence(lm_trans_y).summary_frame()
        influence = influence[["cooks_d", "standard_resid", "hat_diag", "student_resid"]]

        k = len(lm_trans_y.params) - 1
        n = len(data)

        leverage_threshold = 3 * (k + 1) / n
        cooks_d_threshold1 = 0.5
        cooks_d_threshold2 = 1

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].scatter(range(n), influence["student_resid"])
        axs[0].axhline(y=3, color='red', linestyle='--')
        axs[0].axhline(y=-3, color='red', linestyle='--')
        axs[0].set_xlabel('Observation')
        axs[0].set_ylabel('Studentized Residuals')
        axs[0].set_title('Studentized Residuals with ±3 Threshold')

        axs[1].scatter(range(n), influence["hat_diag"])
        axs[1].axhline(y=leverage_threshold, color='red', linestyle='--')
        axs[1].set_xlabel('Observation')
        axs[1].set_ylabel('Leverage')
        axs[1].set_title('Leverage Values with Threshold')

        axs[2].stem(influence["cooks_d"])
        axs[2].axhline(y=cooks_d_threshold1, color='orange', linestyle='--')
        axs[2].axhline(y=cooks_d_threshold2, color='red', linestyle='--')
        axs[2].set_xlabel('Observation')
        axs[2].set_ylabel("Cook's Distance")
        axs[2].set_title("Cook's Distance with Thresholds")

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Influence Analysis: {e}")
        st.stop()


# --------------------------------------------
# Models Tab
# --------------------------------------------

with models_tab:
    st.header("Predictive Modeling Results :computer:", anchor="models")

    # Predictive Modeling
    st.subheader("Predictive Modeling")
    try:
        lm = smf.ols("Happiness_score ~ GDP_per_capita + Social_support + Healthy_life_expectancy + Freedom + Generosity + Perceptions_of_corruption", data=data).fit()
        st.text(lm.summary())

        residuals = lm.resid
        fitted = lm.fittedvalues

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)

        sm.qqplot(residuals, line='s', ax=axs[0, 0])
        axs[0, 0].set_title('QQ Plot of Residuals')

        axs[0, 1].scatter(fitted, residuals)
        axs[0, 1].axhline(y=0, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Fitted Values')
        axs[0, 1].set_ylabel('Residuals')
        axs[0, 1].set_title('Residuals vs Fitted Values')

        axs[1, 0].hist(residuals, bins=15, edgecolor='black')
        axs[1, 0].set_title('Histogram of Residuals')

        sns.boxplot(x=residuals, ax=axs[1, 1])
        axs[1, 1].set_title('Boxplot of Residuals')

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Predictive Modeling: {e}")

    # Predictive Modeling - Transformed X
    st.subheader("Predictive Modeling - Transformed X")
    try:
        lm_trans_x = smf.ols(
            "Happiness_score ~  np.log(GDP_per_capita + 1e-10) + np.log(Social_support + 1e-10) + np.log(Healthy_life_expectancy + 1e-10)",
            data=data).fit()
        st.text(lm_trans_x.summary())

        residuals = lm_trans_x.resid
        fitted = lm_trans_x.fittedvalues

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)

        sm.qqplot(residuals, line='s', ax=axs[0, 0])
        axs[0, 0].set_title('QQ Plot of Residuals')

        axs[0, 1].scatter(fitted, residuals)
        axs[0, 1].axhline(y=0, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Fitted Values')
        axs[0, 1].set_ylabel('Residuals')
        axs[0, 1].set_title('Residuals vs Fitted Values')

        axs[1, 0].hist(residuals, bins=15, edgecolor='black')
        axs[1, 0].set_title('Histogram of Residuals')

        sns.boxplot(x=residuals, ax=axs[1, 1])
        axs[1, 1].set_title('Boxplot of Residuals')

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Predictive Modeling - Transformed X: {e}")

    # Predictive Modeling - Transformed Y
    st.subheader("Predictive Modeling - Transformed Y")
    try:
        lm_trans_y = smf.ols(
            "np.log(Happiness_score) ~ GDP_per_capita + Social_support + Healthy_life_expectancy",
            data=data).fit()
        st.text(lm_trans_y.summary())

        residuals = lm_trans_y.resid
        fitted = lm_trans_y.fittedvalues

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)

        sm.qqplot(residuals, line='s', ax=axs[0, 0])
        axs[0, 0].set_title('QQ Plot of Residuals')

        axs[0, 1].scatter(fitted, residuals)
        axs[0, 1].axhline(y=0, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Fitted Values')
        axs[0, 1].set_ylabel('Residuals')
        axs[0, 1].set_title('Residuals vs Fitted Values')

        axs[1, 0].hist(residuals, bins=15, edgecolor='black')
        axs[1, 0].set_title('Histogram of Residuals')

        sns.boxplot(x=residuals, ax=axs[1, 1])
        axs[1, 1].set_title('Boxplot of Residuals')

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Predictive Modeling - Transformed Y: {e}")

    # Predictive Modeling - Transformed X and Y
    st.subheader("Predictive Modeling - Transformed X and Y")
    try:
        lmv2 = smf.ols(
            "np.log(Happiness_score) ~  np.log(GDP_per_capita + 1e-10) + np.log(Social_support + 1e-10) + np.log(Healthy_life_expectancy + 1e-10)",
            data=data).fit()
        st.text(lmv2.summary())

        residuals = lmv2.resid
        fitted = lmv2.fittedvalues

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)

        sm.qqplot(residuals, line='s', ax=axs[0, 0])
        axs[0, 0].set_title('QQ Plot of Residuals')

        axs[0, 1].scatter(fitted, residuals)
        axs[0, 1].axhline(y=0, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Fitted Values')
        axs[0, 1].set_ylabel('Residuals')
        axs[0, 1].set_title('Residuals vs Fitted Values')

        axs[1, 0].hist(residuals, bins=15, edgecolor='black')
        axs[1, 0].set_title('Histogram of Residuals')

        sns.boxplot(x=residuals, ax=axs[1, 1])
        axs[1, 1].set_title('Boxplot of Residuals')

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Predictive Modeling - Transformed X and Y: {e}")


# --------------------------------------------
# Footer
# --------------------------------------------
st.markdown("---")
st.markdown("Created by Ali Mohamed - Mina Thabet - Berbara Romany - Abdelrahman Mahmoud :computer:")