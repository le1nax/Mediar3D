import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data (already corrected)
data = {
    'Size': [64, 128, 512, 800],
    'cpsam': [109, 265, 2547, 7124],
    'mediar': [21, 28, 182, 1955],
    'hiera': [4, 9, 2231, 9788]
}

df = pd.DataFrame(data)
df_long = df.melt(id_vars='Size', var_name='Network', value_name='Runtime (s)')
df_long['Runtime (min)'] = df_long['Runtime (s)'] / 60

# Map original network names to new names for legend
network_name_map = {
    'cpsam': 'CellposeSAM',
    'mediar': 'Mediar3D',
    'hiera': 'Mediar + SAM2 encoder'
}
df_long['Network_Label'] = df_long['Network'].map(network_name_map)

# Annotation formatting: only show minutes for > 60s
def format_time(row):
    seconds = row['Runtime (s)']
    minutes = row['Runtime (min)']
    if seconds > 60:
        return f'{minutes:.1f} min'
    else:
        return f'{seconds:.0f} s'

df_long['Display Time'] = df_long.apply(format_time, axis=1)
df_long['Size_Str'] = df_long['Size'].astype(str)

# Plotting
plt.figure(figsize=(10, 7))
ax = plt.gca()

# Define markers (different shapes) and solid line styles
markers = ['o', 's', '^']
linestyles = ['-', '-', '-']

networks = df_long['Network'].unique()

# Plot each network separately
for i, network in enumerate(networks):
    subset = df_long[df_long['Network'] == network]
    plt.plot(
        subset['Size_Str'],
        subset['Runtime (s)'],
        label=network_name_map[network],
        marker=markers[i],
        linestyle=linestyles[i],
        markersize=8
    )

    # Annotation logic:
    for _, row in subset.iterrows():
        offset = (0, 10) # Default: above the point
        # Specific overrides for placing annotations below
        if row['Network'] == 'hiera' and row['Size'] in [512, 800]:
            offset = (0, -15) # For hiera 512, 800: below
        elif row['Network'] == 'cpsam' and row['Size'] == 800:
            offset = (0, -15) # NEW: For cpsam 800: below

        plt.annotate(
            row['Display Time'],
            (row['Size_Str'], row['Runtime (s)']),
            textcoords="offset points",
            xytext=offset,
            ha='center',
            fontsize=12
        )

# --- Aesthetic Settings (retained) ---
TICK_FONT_SIZE = 14
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

AXIS_LINE_WIDTH = 2.5
ax.spines['bottom'].set_linewidth(AXIS_LINE_WIDTH)
ax.spines['left'].set_linewidth(AXIS_LINE_WIDTH)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_linewidth(AXIS_LINE_WIDTH)
ax.spines['right'].set_linewidth(AXIS_LINE_WIDTH)
ax.tick_params(width=AXIS_LINE_WIDTH, length=10)

plt.yscale('log')
plt.title('')
plt.xlabel('Image Size (N)', fontsize=16)
plt.ylabel('Runtime (seconds)', fontsize=16)
plt.legend(title='Network', fontsize=14, title_fontsize=14)
plt.grid(True, which="both", ls="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('network_runtime_comparison_cpsam_800_annotation_below.png')
plt.show()
print("Plot generated with 'CellposeSAM' (blue curve) N=800 annotation placed below the point.")