import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -------------------------------
# Settings
# -------------------------------
area_color = '#6baa91'
area_edge = '#000000'
score_color = '#5f86d6'
score_edge = '#000000'
length_color = "#B97878"
length_edge = "#000000"
linewidth = 0.5
alpha = 1
min_area = -0.1
offset = 3700

datasets = ['54MRL', '63MR', '187FN', '203MN', '204FR', '206FRL', '211MRR', '218MN', '21ML', '221ML']

# -------------------------------
# Load data
# -------------------------------
data_dict = {}
for name in datasets:
    data_dict[name] = np.load(
        f'/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/{name}/{name}_all_data.npy',
        allow_pickle=True).item()

# -------------------------------
# Function to process dataset
# -------------------------------
def process_data(data, min_area, offset):
    scores, areas, durations = [], [], []
    for i, video in enumerate(data):
        last_end = -1
        for interval, values in sorted(data[video].items()):
            start, end = interval[0] + offset * i, interval[1] + offset * i
            if values['area'] > min_area and start >= last_end:
                scores.append(values['ratio'])
                areas.append(values['area'])
                durations.append(values['duration'] / 13)
                last_end = end
    return scores, areas, durations

# -------------------------------
# Process all datasets
# -------------------------------
scores_all, areas_all, lengths_all = [], [], []
for name in datasets:
    s, a, l = process_data(data_dict[name], min_area, offset)
    scores_all.append(s)
    areas_all.append(a)
    lengths_all.append(l)

# -------------------------------
# Histogram bins
# -------------------------------
bin_list = [
    np.linspace(0, 1, 11),  # scores
    np.linspace(0, 1, 11),  # areas
    np.linspace(0, 15, 16)  # durations
]

colors = [score_color, area_color, length_color]
edge_colors = [score_edge, area_edge, length_edge]

# -------------------------------
# Create figure
# -------------------------------
figsize_cm = (18, 18)
figsize_in = tuple(x / 2.54 for x in figsize_cm)
fig = plt.figure(figsize=figsize_in)
outer_gs = gridspec.GridSpec(nrows=5, ncols=6, figure=fig, hspace=0.75, wspace=0.4)

label_size = 8
# -------------------------------
# Plot loop
# -------------------------------
for i in range(5):  # 5 rows
    for j in range(2):  # 2 datasets per row
        idx = i*2 + j
        if idx >= len(datasets):
            continue

        # Column positions: 3 columns per dataset
        col_start = j*3
        name = datasets[idx]

        # Waviness histogram
        ax_score = fig.add_subplot(outer_gs[i, col_start])
        counts, bins_, patches = ax_score.hist(scores_all[idx], bins=bin_list[0], color=colors[0],
                                              edgecolor=edge_colors[0], alpha=alpha,rwidth = 1, linewidth = linewidth)
        percentage = (counts / len(scores_all[idx]))*100
        for count, patch in zip(percentage, patches):
            patch.set_height(count)
        ax_score.set_ylim([0, 100])
        ax_score.set_title(f"{name} , N={len(scores_all[idx])}", fontsize=8)
        ax_score.spines['top'].set_visible(False)
        ax_score.spines['right'].set_visible(False)
        ax_score.tick_params(axis='x', labelsize=label_size)
        ax_score.tick_params(axis='y', labelsize=label_size)
        ax_score.set_xticks([0,0.5, 1])
        ax_score.set_xticklabels(['0', '0.5', '1'])
        ax_score.set_yticks([0, 50 , 100])
        ax_score.set_yticklabels(["0", "50", "100"])

        # Area histogram
        ax_area = fig.add_subplot(outer_gs[i, col_start+1])
        counts, bins_, patches = ax_area.hist(areas_all[idx], bins=bin_list[1], color=colors[1],
                                              edgecolor=edge_colors[1], alpha=alpha, rwidth = 1, linewidth = linewidth)
        percentage = (counts / len(areas_all[idx]))*100
        for count, patch in zip(percentage, patches):
            patch.set_height(count)
        ax_area.set_ylim([0, 40])
        ax_area.spines['top'].set_visible(False)
        ax_area.spines['right'].set_visible(False)
        ax_area.tick_params(axis='x', labelsize=label_size)
        ax_area.tick_params(axis='y', labelsize=label_size)
        ax_area.set_xticks([0, 0.5, 1])
        ax_area.set_xticklabels(['0', '0.5', '1'])
        ax_area.set_yticks([0, 20 , 40])
        ax_area.set_yticklabels(["0", "20", "40"])

        # Duration histogram
        ax_length = fig.add_subplot(outer_gs[i, col_start+2])
        counts, bins_, patches = ax_length.hist(lengths_all[idx], bins=bin_list[2], color=colors[2],
                                                edgecolor=edge_colors[2], alpha=alpha, rwidth = 1, linewidth = linewidth)

        percentage = (counts / len(lengths_all[idx]))*100
        for count, patch in zip(percentage, patches):
            patch.set_height(count)
        ax_length.set_ylim([0, 30])
        ax_length.spines['top'].set_visible(False)
        ax_length.spines['right'].set_visible(False)
        ax_length.tick_params(axis='x', labelsize=label_size)
        ax_length.tick_params(axis='y', labelsize=label_size)
        ax_length.set_xticks([0, 5, 10, 15])
        ax_length.set_xticklabels(["0", "5", "10", "15"])
        ax_length.set_yticks([0, 15, 30])
        ax_length.set_yticklabels(["0", "15", "30"])

        ax_score.set_xlim(0, 1)
        ax_area.set_xlim(0, 1)
        ax_length.set_xlim(0, 15)

        if j == 0:
            if i==0:
                ax_score.set_ylabel("Events [%]", fontsize=8)
            else:
                ax_score.set_ylabel("%", fontsize=8)
           # ax_area.set_ylabel("%", fontsize=8)
           # ax_length.set_ylabel("%", fontsize=8)
        if i == 4:
            ax_score.set_xlabel("Waviness", fontsize=8)
            ax_area.set_xlabel("Active Area", fontsize=8)
            ax_length.set_xlabel("Length [sec]", fontsize=8)

# -------------------------------
# Layout
# -------------------------------
plt.tight_layout()
plt.savefig("Figure_All_Datasets_5x6.pdf", dpi=1000)
plt.show()
