
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA
# =========================

data = [
    ["Fatih", "Peak", 1.65, 0.36, 1.65, 0.36, 1],
    ["emaan2", "Foot", 7.98, 1.45, 7.98, 1.45, 1],
    ["emaan2", "Foot", 10.61, 1.93, 10.61, 1.93, 1],
    ["Zehra", "Peak", 0.20, 3.30, 0.20, 3.30, 1],
    ["Silan", "Peak", 1.06, 3.53, 1.06, 3.53, 1],
    ["Silan", "Peak", 0.35, 2.25, 0.35, 2.25, 1],
    ["Silan", "Peak", 1.41, 1.57, 1.41, 1.57, 1],
    ["emaan2", "Peak", 11.45, 7.12, 11.45, 7.12, 1],
    ["dfgh2", "Peak", 3.33, 0.78, 3.68, 1.19, 119],
    ["emaan", "Peak", 6.23, 4.29, 6.23, 4.29, 1],
    ["alize", "Foot", 0.43, 0.67, 0.43, 0.67, 1],
    ["alize", "Peak", 0.29, 3.20, 0.29, 3.20, 1],
]

df = pd.DataFrame(data, columns=[
    "User", "Mode", "MAE_SBP", "MAE_DBP", "RMSE_SBP", "RMSE_DBP", "Windows"
])

# =========================
# 1. OVERALL SUMMARY GRAPH
# =========================

metrics = ["MAE SBP", "MAE DBP", "RMSE SBP", "RMSE DBP"]
means = [3.749, 2.538, 3.778, 2.572]
medians = [1.530, 2.090, 1.530, 2.090]
stds = [4.209, 1.906, 4.207, 1.875]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, means, width, label="Mean")
plt.bar(x, medians, width, label="Median")
plt.bar(x + width, stds, width, label="Std Dev")
plt.xticks(x, metrics)
plt.ylabel("Error (mmHg)")
plt.title("Overall Blood Pressure Error Statistics")
plt.legend()
plt.tight_layout()
plt.savefig("img/01_overall_summary_statistics.png", dpi=300)
plt.show()

# =========================
# 2. MODE PERFORMANCE GRAPH
# =========================

modes = ["Foot SBP", "Foot DBP", "Peak SBP", "Peak DBP"]
mean_errors = [6.340, 1.350, 2.885, 2.933]
median_errors = [7.980, 1.450, 1.410, 3.200]

x = np.arange(len(modes))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, mean_errors, width, label="Mean Error")
plt.bar(x + width / 2, median_errors, width, label="Median Error")
plt.xticks(x, modes)
plt.ylabel("Error (mmHg)")
plt.title("Mode-Based Blood Pressure Tracking Performance")
plt.legend()
plt.tight_layout()
plt.savefig("img/02_mode_performance_comparison.png", dpi=300)
plt.show()

# =========================
# 3. SBP VS DBP CORRELATION
# =========================

correlation = df["MAE_SBP"].corr(df["MAE_DBP"])

plt.figure(figsize=(8, 6))
plt.scatter(df["MAE_SBP"], df["MAE_DBP"])

for i, row in df.iterrows():
    plt.text(row["MAE_SBP"], row["MAE_DBP"], row["User"], fontsize=8)

plt.xlabel("SBP Error (mmHg)")
plt.ylabel("DBP Error (mmHg)")
plt.title(f"SBP vs DBP Error Correlation (r = {correlation:.3f})")
plt.tight_layout()
plt.savefig("img/03_sbp_dbp_correlation.png", dpi=300)
plt.show()

# =========================
# 4. DONUT CHART
# =========================

labels = ["SBP Mean Error", "DBP Mean Error"]
sizes = [df["MAE_SBP"].mean(), df["MAE_DBP"].mean()]

plt.figure(figsize=(7, 7))
plt.pie(
    sizes,
    labels=labels,
    autopct="%1.2f mmHg",
    wedgeprops={"width": 0.4}
)
plt.title("Average Blood Pressure Error Distribution")
plt.tight_layout()
plt.savefig("img/04_donut_error_distribution.png", dpi=300)
plt.show()

# =========================
# 5. BOXPLOT
# =========================

plt.figure(figsize=(8, 6))
plt.boxplot(
    [df["MAE_SBP"], df["MAE_DBP"]],
    labels=["SBP Error", "DBP Error"]
)
plt.ylabel("Error (mmHg)")
plt.title("Distribution and Variability of BP Errors")
plt.tight_layout()
plt.savefig("img/05_bp_error_boxplot.png", dpi=300)
plt.show()

# =========================
# 6. MAE PER RECORDING
# =========================

x = np.arange(len(df))
width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, df["MAE_SBP"], width, label="SBP MAE")
plt.bar(x + width / 2, df["MAE_DBP"], width, label="DBP MAE")
plt.xticks(x, df["User"], rotation=45)
plt.ylabel("Error (mmHg)")
plt.title("SBP vs DBP Mean Absolute Error per Recording")
plt.legend()
plt.tight_layout()
plt.savefig("img/06_mae_per_recording.png", dpi=300)
plt.show()

# =========================
# 7. RMSE PER RECORDING
# =========================

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, df["RMSE_SBP"], width, label="SBP RMSE")
plt.bar(x + width / 2, df["RMSE_DBP"], width, label="DBP RMSE")
plt.xticks(x, df["User"], rotation=45)
plt.ylabel("RMSE (mmHg)")
plt.title("SBP vs DBP RMSE per Recording")
plt.legend()
plt.tight_layout()
plt.savefig("img/07_rmse_per_recording.png", dpi=300)
plt.show()

# =========================
# PRINT STATISTICS
# =========================

print("\nOverall Statistics:")
print(df[["MAE_SBP", "MAE_DBP", "RMSE_SBP", "RMSE_DBP"]].describe())

print("\nMode-Based Mean Errors:")
print(df.groupby("Mode")[["MAE_SBP", "MAE_DBP"]].mean())

print("\nCorrelation between SBP and DBP MAE:")
print(correlation)

print("\nGraphs saved successfully as PNG files.")