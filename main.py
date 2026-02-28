import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import cleaner

@st.cache_data
def load_data():
    df = pd.read_csv("./data/output.csv")
    return df


def handle_task_1(df):
    st.header("Task 1: Dimensionality Reduction (PCA)")
    st.write("Project PM2.5, PM10, NO2, and O3 into 2D space and analyze variable loadings.")

    variables = ["pm25", "pm10", "no2", "o3"]

    df_pca = df.dropna(subset=variables)

    X = df_pca[variables]


    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    df_pca["PC1"] = components[:, 0]
    df_pca["PC2"] = components[:, 1]

    st.subheader("Explained Variance Ratio")
    st.write(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5, s=10)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Projection of Pollution Variables")
    st.pyplot(fig)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=variables
    )

    st.subheader("PCA Loadings (Variable Contributions)")
    st.dataframe(loadings)

    st.markdown("""
    **Interpretation:**
    - Variables with larger absolute loadings contribute more to each principal component.
    - PC1 typically captures overall pollution intensity.
    - PC2 often captures contrast between gas pollutants (NO2, O3) and particulate matter (PM2.5, PM10).
    """)

def handle_task_2(df):
    st.header("Task 2: High-Density Temporal Analysis")
    st.write("Health Threshold Violations (PM2.5 > 35 μg/m³)")

    df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"])

    pm25_df = df[["location_id", "datetimeUtc", "pm25_raw"]].dropna()

    pm25_df["violation"] = (pm25_df["pm25_raw"] > 35).astype(int)

    pm25_df["date"] = pm25_df["datetimeUtc"].dt.date
    daily = pm25_df.groupby(["location_id", "date"])["violation"].mean().reset_index()

    heatmap_data = daily.pivot_table(
        index="location_id",
        columns="date",
        values="violation",
        fill_value=0
    )

    st.subheader("PM2.5 Threshold Violation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        cmap="Reds",
        cbar_kws={"label": "Violation Frequency"},
        ax=ax
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sensor (Location ID)")
    ax.set_title("Daily PM2.5 Health Threshold Violations Across Sensors")

    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**
    - Vertical bands → city-wide pollution events  
    - Horizontal bands → consistently polluted neighborhoods  
    - Repeating patterns → daily (24-hour) traffic cycles  
    - Slow shifts → seasonal/monthly patterns  
    """)
    

def handle_task_3(df):
    st.header("Task 3: Distribution Modeling & Tail Integrity")
    st.write("Analyze extreme PM2.5 pollution events (PM2.5 > 200 μg/m³) in an industrial zone.")

    # Ensure pm2.5_raw exists
    if "pm25_raw" not in df.columns:
        st.error("Column pm2.5_raw not found. Ensure preprocessing pipeline created raw values.")
        return

    # Filter for a selected industrial zone
    # Assuming you have a 'zone' column (industrial/residential)
    if "zone" in df.columns:
        df_industrial = df[df["zone"] == "industrial"]
    else:
        st.warning("No 'zone' column found. Using all data for demonstration.")
        df_industrial = df

    pm25 = df_industrial["pm25_raw"].dropna()

    # Compute 99th percentile
    p99 = pm25.quantile(0.99)
    st.subheader(f"99th Percentile of PM2.5: {p99:.2f} μg/m³")

    st.subheader("Peak-Focused Histogram")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    sns.histplot(pm25, bins=50, kde=False, color="skyblue", ax=ax1)
    ax1.set_xlabel("pm25")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Histogram Highlighting Peaks (Most Common Values)")
    st.pyplot(fig1)

    # Tail-focused histogram: only extreme PM2.5
    tail_pm25 = pm25[pm25 > 100]  # adjust threshold to include long tail
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.histplot(tail_pm25, bins=30, kde=False, color="salmon", ax=ax2, log_scale=(False, True))
    ax2.set_xlabel("PM2.5 (μg/m³)")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title("Histogram Highlighting Long Tail of Extreme Values")
    st.pyplot(fig2)
    st.markdown("""
    **Interpretation:**
    - Peak-focused histogram shows where most measurements lie (normal urban pollution range).  
    - Tail-focused histogram emphasizes rare, extreme events that may exceed hazardous thresholds.  
    - The 99th percentile helps quantify “Extreme Hazard” levels and assess risk.
    """)


def main():
    st.set_page_config(page_title="Urban Environmental Intelligence Dashboard", layout="wide")
    
    st.title("Urban Environmental Intelligence Challenge Dashboard")
    
    df = load_data()
    
    tab1, tab2, tab3 = st.tabs([
        "Task 1: Dimensionality Reduction",
        "Task 2: Temporal Analysis",
        "Task 3: Distribution & Tails",
    ])
    
    with tab1:
        handle_task_1(df)
        
    with tab2:
        handle_task_2(df)
        
    with tab3:
        handle_task_3(df)


if __name__ == "__main__":
    #cleaner.run_data_cleaning_pipeline()
    main()