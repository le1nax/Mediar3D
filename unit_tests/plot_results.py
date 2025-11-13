import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/Hiera/summary_boxplots/summary_seg_boxplot_iou_mean.csv')

# Rename columns for clarity based on your request
df.rename(columns={
    'SEG_min': 'Lower_SEG',
    'SEG_median': 'Median_SEG',
    'SEG_max': 'Highest_SEG'
}, inplace=True)

# Select the required columns and unique ROIs
plot_data = df[['ROI', 'Threshold', 'Lower_SEG', 'Median_SEG', 'Highest_SEG']]
unique_rois = plot_data['ROI'].unique()

# --- Visualization Parameters (Inspired by the image) ---
# Define colors for different ROIs (you can expand this list)
colors = plt.cm.get_cmap('tab10', len(unique_rois))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate through each unique ROI to plot its curves
for i, roi in enumerate(unique_rois):
    roi_data = plot_data[plot_data['ROI'] == roi].sort_values(by='Threshold')
    color = colors(i)
    
    # 1. Plot the vertical bar connecting Lower_SEG and Highest_SEG
    # The 'yerr' argument is typically used for symmetric errors; 
    # we'll use a custom loop or ax.vlines/ax.plot for asymmetric min/max lines 
    # as lines without markers for the 'bars'.
    
    # Calculate the 'error' from the median for the plot (used for error bar style, but here we'll use ax.vlines for pure line)
    # y_min = roi_data['Lower_SEG'].values
    # y_max = roi_data['Highest_SEG'].values
    # y_median = roi_data['Median_SEG'].values
    # x_threshold = roi_data['Threshold'].values

    # Plot a vertical line segment for each threshold point
    # Note: Using ax.vlines creates vertical segments more cleanly than ax.errorbar 
    # for a min-to-max connector without a marker.
    for x, y_low, y_high in zip(roi_data['Threshold'], roi_data['Lower_SEG'], roi_data['Highest_SEG']):
        ax.plot([x, x], [y_low, y_high], linestyle='-', color=color, alpha=0.5, linewidth=1)
        # Add a small dash at the min/max (optional, but good practice for boxplots)
        ax.plot(x, y_low, marker='_', color=color, markersize=8)
        ax.plot(x, y_high, marker='_', color=color, markersize=8)

    # 2. Plot the graph line through the Median_SEG
    ax.plot(
        roi_data['Threshold'], 
        roi_data['Median_SEG'], 
        label=f'Median SEG ({roi})', 
        color=color, 
        marker='o', # Use a marker for points
        linestyle='-',
        linewidth=2
    )
    
# --- Axis Configuration (Inspired by the image) ---

# Set labels and title
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Segmentation Score (SEG)', fontsize=12)
ax.set_title('Segmentation Score vs. Threshold by ROI', fontsize=14)

# Set X-axis ticks (assuming Threshold values are 0.1, 0.3, 0.5, 0.7, 0.9 based on snippet)
ax.set_xticks(sorted(df['Threshold'].unique()))

# Set Y-axis limits (The image seems to range from ~0.5 to ~1.0)
min_y = plot_data['Lower_SEG'].min()
max_y = plot_data['Highest_SEG'].max()
ax.set_ylim(max(0.0, min_y - 0.05), min(1.0, max_y + 0.05)) # Add small padding

# Add grid (as seen in the image)
ax.grid(True, linestyle='--', alpha=0.6)

# Add legend
ax.legend(title='ROI', loc='lower right')

# Display the plot
plt.tight_layout()
plt.show()

# Print confirmation
print(f"Chart generated for {len(unique_rois)} ROIs, showing Lower, Median, and Highest SEG scores over Threshold.")