import pandas as pd
import matplotlib.pyplot as plt

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

# 1. SBP and DBP MAE per recording
plt.figure(figsize=(12, 6))
x = range(len(df))
plt.bar(x, df["MAE_SBP"], width=0.4, label="SBP MAE")
plt.bar([i + 0.4 for i in x], df["MAE_DBP"], width=0.4, label="DBP MAE")
plt.xticks([i + 0.2 for i in x], df["User"], rotation=45)
plt.ylabel("Error (mmHg)")
plt.title("SBP vs DBP Mean Absolute Error per Recording")
plt.legend()
plt.tight_layout()
plt.savefig("mae_per_recording.png", dpi=300)
plt.show()

# 2. Error distribution
plt.figure(figsize=(8, 5))
plt.hist(df["MAE_SBP"], bins=6, alpha=0.7, label="SBP MAE")
plt.hist(df["MAE_DBP"], bins=6, alpha=0.7, label="DBP MAE")
plt.xlabel("Error (mmHg)")
plt.ylabel("Frequency")
plt.title("Distribution of BP Errors")
plt.legend()
plt.tight_layout()
plt.savefig("error_distribution.png", dpi=300)
plt.show()

# 3. Peak vs Foot comparison
mode_mean = df.groupby("Mode")[["MAE_SBP", "MAE_DBP"]].mean()

mode_mean.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Mean Error (mmHg)")
plt.title("Average Error by Detection Mode")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("mode_comparison.png", dpi=300)
plt.show()

# 4. SBP vs DBP correlation
plt.figure(figsize=(7, 5))
plt.scatter(df["MAE_SBP"], df["MAE_DBP"])

for i, row in df.iterrows():
    plt.text(row["MAE_SBP"], row["MAE_DBP"], row["User"], fontsize=8)

plt.xlabel("SBP MAE (mmHg)")
plt.ylabel("DBP MAE (mmHg)")
plt.title("Correlation Between SBP and DBP Error")
plt.tight_layout()
plt.savefig("sbp_dbp_correlation.png", dpi=300)
plt.show()

# 5. RMSE comparison
plt.figure(figsize=(12, 6))
plt.bar(x, df["RMSE_SBP"], width=0.4, label="SBP RMSE")
plt.bar([i + 0.4 for i in x], df["RMSE_DBP"], width=0.4, label="DBP RMSE")
plt.xticks([i + 0.2 for i in x], df["User"], rotation=45)
plt.ylabel("RMSE (mmHg)")
plt.title("SBP vs DBP RMSE per Recording")
plt.legend()
plt.tight_layout()
plt.savefig("rmse_per_recording.png", dpi=300)
plt.show()

print("Overall statistics:")
print(df[["MAE_SBP", "MAE_DBP", "RMSE_SBP", "RMSE_DBP"]].describe())

print("\nMode-based mean errors:")
print(mode_mean)

print("\nCorrelation SBP vs DBP MAE:")
print(df["MAE_SBP"].corr(df["MAE_DBP"]))
