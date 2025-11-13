import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Load the specified CSV file
file_name = '/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_small_results/paper/summary_boxplot/summary_seg_boxplot_iou_mean.csv' 
df = pd.read_csv(file_name)

# # --- START: Code for merging two files ---

# # Define the file names. Use the full path for the first file, or just the
# # filename if it's already in the current working directory.
# file_name_1 = '/work/scratch/geiger/Datasets/Blasto/final_test/final_results/paper/summary/summary_seg_boxplot_iou_mean_paper1.csv'
# file_name_2 = '/work/scratch/geiger/Datasets/Blasto/final_test/final_results/paper_blast_lowertresh/summary_boxplot/summary_seg_boxplot_iou_mean.csv' 

# # Load the DataFrames
# df1 = pd.read_csv(file_name_1)
# df2 = pd.read_csv(file_name_2)

# # Combine the dataframes. `ignore_index=True` ensures the new DataFrame has
# # a continuous index, which is safe for concatenation.
# df = pd.concat([df1, df2], ignore_index=True)

# --- END: Code for merging two files ---

# Rename columns for clarity
df.rename(columns={
    'SEG_min': 'Lower_SEG',
    'SEG_median': 'Median_SEG',
    'SEG_max': 'Highest_SEG'
}, inplace=True)

# Define the target ROIs and the model filter (Hiera for SAM2 Encoder)
# These are the ROIs present in the Hiera data: 100, 1000, 10000.
target_rois = ['ROI_0', 'ROI_100', 'ROI_100000']
target_rois = ['ROI_0', 'ROI_100', 'ROI_100000']
model_filter = 'Hiera' 
plot_rois_str = ['0', '100', '100000'] # Order for plotting/color assignment
#df['ROI'] = df['ROI'].astype(str).apply(lambda x: f'ROI_{x}')
# Filter the DataFrame to the target ROIs AND models containing 'Hiera'
plot_data = df[
    df['ROI'].isin(target_rois)# & df['Model'].str.contains(model_filter, na=False)

].copy()

# Extract number part for labeling
plot_data['ROI_str'] = plot_data['ROI'].str.split('_').str[-1] 

# --- Offset Calculation for Non-Overlapping Bars (3 series) ---
# Reduced offset for narrower spacing between bars
LOG_OFFSET = 0.05 

# Map ROI string to its offset for the 100, 1000, 10000 series
OFFSET_MAP = {
    '100000': -LOG_OFFSET, # Largest ROI -> Shift Left
    '100': 0.0,           # Middle ROI -> No Shift
    '0': LOG_OFFSET      # Smallest ROI -> Shift Right
}

def get_shifted_x(threshold, roi_str):
    """Calculates the shifted x-position for plotting on a log scale."""
    offset = OFFSET_MAP.get(roi_str, 0.0)
    # Apply offset in log space and convert back to linear scale
    return 10**(np.log10(threshold) + offset)

# Apply the shifted X-position to the data
plot_data['Shifted_Threshold'] = plot_data.apply(
    lambda row: get_shifted_x(row['Threshold'], row['ROI_str']), 
    axis=1
)

# Get all original, unshifted threshold values for the ticks
all_threshold_values = sorted(plot_data['Threshold'].unique())


# --- Visualization Parameters ---
plt.rcParams['font.family'] = 'serif'
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
BAR_LINEWIDTH = 2.5 
MEDIAN_LINEWIDTH = 2
TITLE_FONTSIZE = 20

# Define the colormap and custom mapping for three ROIs:
colors_map = plt.cm.get_cmap('Dark2', 3)
roi_color_indices = {
    '100000': 0,  # Largest ROI (Cyan/Blue) -> index 0 
    '100': 1,   # Middle ROI (Orange/Yellow) -> index 1
    '0': 2       # Smallest ROI (Green/Gray) -> index 2
}

# Create the plot, making it square: (6, 6)
fig, ax = plt.subplots(figsize=(6, 6))

# Set the headline 
ax.set_title('Mediar + Paper Weights', fontsize=TITLE_FONTSIZE, pad=10) 
#ax.set_title('Mediar + Extended Pretraining', fontsize=TITLE_FONTSIZE, pad=10) 
#ax.set_title('Mediar + SAM2 Encoder', fontsize=TITLE_FONTSIZE, pad=10) 

# Iterate through each unique ROI to plot its curves
for roi_str in plot_rois_str:
    if roi_str in plot_data['ROI_str'].unique():
        # Sort by the ORIGINAL Threshold for correct line plotting order
        roi_data = plot_data[plot_data['ROI_str'] == roi_str].sort_values(by='Threshold')
        
        # Get the assigned color
        color = colors_map(roi_color_indices[roi_str])
        
        # Define the legend label
        legend_label = f'Median SEG ({roi_str} ROIs)'
        
        # 1. Plot the vertical bar connecting Lower_SEG and Highest_SEG 
        # Use the Shifted_Threshold for the bar position
        for x_shifted, y_low, y_high in zip(roi_data['Shifted_Threshold'], roi_data['Lower_SEG'], roi_data['Highest_SEG']):
            ax.plot([x_shifted, x_shifted], [y_low, y_high], linestyle='-', color=color, alpha=0.7, linewidth=BAR_LINEWIDTH)

        # 2. Plot the graph line through the Median_SEG
        # Use the Shifted_Threshold for the line position
        ax.plot(
            roi_data['Shifted_Threshold'], 
            roi_data['Median_SEG'], 
            label=legend_label, 
            color=color, 
            marker='o', 
            linestyle='-',
            linewidth=MEDIAN_LINEWIDTH
        )
    
# --- Axis Configuration ---
ax.set_xlabel('Cellprobability Threshold', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel('SEG-Score', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_xscale('log')

exclude_vals = [0.9, 0.7, 0.3, 0.005]  # all values you want to skip
threshold_values_for_ticks = [t for t in all_threshold_values if not any(np.isclose(t, x) for x in exclude_vals)]

ax.set_xticks(threshold_values_for_ticks)

# Define the custom formatter for 10^x notation and decimal numbers
def custom_log_formatter(x, pos):
    if x < 0.1 and x > 0 and (np.log10(x) % 1 == 0 or np.isclose(np.log10(x) % 1, 0)):
        # Use scientific notation for exact powers of 10
        return f'${{10^{{{np.log10(x):.0f}}}}}$'
    # Use simple decimal notation for all other values
    return f'{x:g}'

# Apply the custom formatter
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(custom_log_formatter))

ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
ax.tick_params(axis='x', rotation=0) 

# Set Y-axis limits 
if not plot_data.empty:
    min_y = plot_data['Lower_SEG'].min()
    max_y = plot_data['Highest_SEG'].max()
    ax.set_ylim(max(0.0, min_y - 0.05), min(1.0, max_y + 0.05))

ax.grid(True, linestyle='--', alpha=0.6, which='major') 

# Add legend - loc='lower left'
ax.legend(title='#ROIs trained', loc='lower left', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE + 2)

# Save the plot
plt.tight_layout()

plt.savefig('/home/students/geiger/Downloads/plots/new_post_processing.png')