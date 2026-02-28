import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


    if "pm25_raw" not in df.columns:
        st.error("Column pm2.5_raw not found. Ensure preprocessing pipeline created raw values.")
        return


    if "zone" in df.columns:
        df_industrial = df[df["zone"] == "industrial"]
    else:
        st.warning("No 'zone' column found. Using all data for demonstration.")
        df_industrial = df

    pm25 = df_industrial["pm25_raw"].dropna()


    p99 = pm25.quantile(0.99)
    st.subheader(f"99th Percentile of PM2.5: {p99:.2f} μg/m³")

    st.subheader("Peak-Focused Histogram")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    sns.histplot(pm25, bins=50, kde=False, color="skyblue", ax=ax1)
    ax1.set_xlabel("pm25")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Histogram Highlighting Peaks (Most Common Values)")
    st.pyplot(fig1)

 
    tail_pm25 = pm25[pm25 > 100]  
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


def handle_task_4(df):
    st.header("Task 4: Visual Integrity Audit")
    st.write("Evaluating the proposal to use a 3D bar chart for Pollution vs. Population Density vs. Region.")

 
    st.subheader("1. Proposal Evaluation — 3D Bar Chart")
    st.markdown("""
    **Decision: REJECT the 3D bar chart proposal.**

    | Principle | Issue with 3D Bar Chart |
    |---|---|
    | **Lie Factor** | Perspective distortion causes bars at the back to appear smaller than bars of equal value at the front, inflating the perceived Lie Factor (visual effect ÷ data effect ≠ 1). |
    | **Data-Ink Ratio** | 3D depth, shadows, and grid planes add substantial non-data ink without conveying additional information, violating Tufte's principle of maximising the data-ink ratio. |

    **Conclusion:** A 3D bar chart misleads viewers and wastes ink.  
    We replace it with a **Small Multiples** approach — one faceted scatter plot per region — which keeps the Lie Factor close to 1 and maximises the data-ink ratio.
    """)


    required_cols = ["region", "population_density", "pm25_raw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Re-run the data cleaning pipeline.")
        return

    agg = (
        df.dropna(subset=required_cols)
        .groupby(["region", "location_id"])
        .agg(
            avg_pm25=("pm25_raw", "mean"),
            population_density=("population_density", "first"),
        )
        .reset_index()
    )


    st.subheader("2. Small Multiples — Pollution vs. Population Density by Region")

    regions = sorted(agg["region"].unique())
    n_regions = len(regions)
    fig, axes = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4), sharey=True)
    if n_regions == 1:
        axes = [axes]


    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(
        vmin=agg["avg_pm25"].min(), vmax=agg["avg_pm25"].max()
    )

    for ax, region in zip(axes, regions):
        subset = agg[agg["region"] == region]
        sc = ax.scatter(
            subset["population_density"],
            subset["avg_pm25"],
            c=subset["avg_pm25"],
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.5,
            s=70,
            alpha=0.85,
        )
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xlabel("Pop. Density\n(people/km²)")
        if ax == axes[0]:
            ax.set_ylabel("Avg PM2.5 (μg/m³)")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label="Avg PM2.5 (μg/m³)",
        shrink=0.8,
    )
    fig.suptitle(
        "Small Multiples: Avg PM2.5 vs Population Density per Region",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Why Small Multiples?**  
    - Each panel shares the same axes, enabling direct comparison across regions.  
    - No perspective distortion → Lie Factor ≈ 1.  
    - Minimal non-data ink → high Data-Ink Ratio.  
    - A third variable (pollution magnitude) is encoded via colour without adding a misleading 3D axis.
    """)


    st.subheader("3. Bivariate Heatmap — Region × Population-Density Bin")

    agg["pop_density_bin"] = pd.cut(
        agg["population_density"], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    heat = agg.pivot_table(
        index="region", columns="pop_density_bin", values="avg_pm25", aggfunc="mean"
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        heat,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Avg PM2.5 (μg/m³)"},
        ax=ax2,
    )
    ax2.set_title("Avg PM2.5 by Region and Population Density Band")
    ax2.set_xlabel("Population Density Band")
    ax2.set_ylabel("Region")
    st.pyplot(fig2)


    st.subheader("4. Color Scale Justification — Sequential vs. Rainbow")
    st.markdown("""
    | Criterion | Sequential (YlOrRd) | Rainbow (jet / hsv) |
    |---|---|---|
    | **Perceptual ordering** | Monotonically increasing luminance → viewers naturally read low-to-high | Non-monotonic luminance creates false boundaries (e.g., cyan ↔ green looks like a step that doesn't exist in data) |
    | **Colour-blind safety** | Degrades gracefully to luminance channel | Reds and greens become indistinguishable for ~8 % of males |
    | **Human luminance perception** | Aligns with the visual system's strongest channel (luminance > hue) | Relies on hue, which humans rank-order poorly |

    **Choice:** We use **YlOrRd** (Yellow → Orange → Red), a sequential colour map that:
    - Preserves data ordering through steadily decreasing luminance.
    - Remains readable when printed in greyscale.
    - Avoids the artificial "bands" that rainbow maps impose on continuous data.
    """)


def main():
    st.set_page_config(page_title="Urban Environmental Intelligence Dashboard", layout="wide")
    
    st.title("Urban Environmental Intelligence Challenge Dashboard")
    
    df = load_data()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Task 1: Dimensionality Reduction",
        "Task 2: Temporal Analysis",
        "Task 3: Distribution & Tails",
        "Task 4: Visual Integrity Audit",
    ])
    
    with tab1:
        handle_task_1(df)
        
    with tab2:
        handle_task_2(df)
        
    with tab3:
        handle_task_3(df)

    with tab4:
        handle_task_4(df)


if __name__ == "__main__":
    #cleaner.run_data_cleaning_pipeline()
    main()