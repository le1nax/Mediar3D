import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load CSV
df = pd.read_csv("/work/scratch/geiger/Datasets/Blasto/final_test/final_results/paper/summary/summary_seg_boxplot_iou_mean.csv")

# Extract epoch number from Model column
df['Epoch'] = df['Model'].apply(lambda x: int(re.search(r'epoch(\d+)', x).group(1)) if re.search(r'epoch(\d+)', x) else 0)

# Optionally aggregate across ROIs
agg_df = df.groupby(['Threshold', 'Epoch']).agg(SEG_median=('SEG_median', 'mean')).reset_index()

# Set seaborn style
sns.set(style="whitegrid")

# Create a subplot per Threshold
thresholds = sorted(agg_df['Threshold'].unique())
fig, axes = plt.subplots(len(thresholds), 1, figsize=(8, 4*len(thresholds)), sharex=True)

if len(thresholds) == 1:
    axes = [axes]

for ax, th in zip(axes, thresholds):
    subset = agg_df[agg_df['Threshold'] == th]
    sns.lineplot(data=subset, x='Epoch', y='SEG_median', marker='o', ax=ax)
    ax.set_title(f"Threshold = {th}")
    ax.set_ylabel("SEG median")
    ax.set_xlabel("Epochs")

plt.tight_layout()
plt.show()