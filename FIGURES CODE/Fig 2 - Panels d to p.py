import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import chain
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import os




score_color = "#5f86d6"
score_edge = '#000000'

area_color = "#6baa91"
area_edge = "#000000"

length_color = "#B97878"
length_edge = "#000000"
linewidth = 0.5

# Create final figure
figsize_cm = (18, 15)
figsize_in = tuple(x / 2.54 for x in figsize_cm)

fig = plt.figure(figsize=figsize_in)
outer_gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[1,1,1,1,1],)

axes = []

# Rows 0,1,2: 1 axis spanning full width
for i in range(3):
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[i])
    ax = fig.add_subplot(inner_gs[0, 0])
    axes.append(ax)



# Rows 4 and 5: 5 columns each, equally sized axes
for row in [3, 4]:
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[row],wspace=0.3)
    for col in range(5):
        ax = fig.add_subplot(inner_gs[0, col])
        axes.append(ax)




base_path = ''
statistics_path = os.path.join( base_path, 'Fig 2 - statistics')



video_name = '218MN'  ### Fig 2 video

data = np.load(os.path.join(statistics_path,f'{video_name}_statistics.npy'),allow_pickle=True).item()

min_area=-0.1
score_data, area_data, duration_data, energy_data, intervals_data = [], [], [], [], []

for i, video in enumerate(data):
    video_intervals = []
    video_scores = []
    video_areas = []

    last_interval = None
    last_values = None

    for interval, values in sorted(data[video].items()):
        start, end = interval[0] + 3700 * i, interval[1] + 3700 * i
        #print(i , start%3700,end%3700)

        if values['area'] <= min_area:
            continue

        if last_interval is None:
            # first interval
            last_interval = (start, end)
            last_values = values
        else:
            # check overlap
            if start < last_interval[1]:
                # intervals overlap → keep the one with the larger area
                if values['area'] > last_values['area']:
                    last_interval = (start, end)
                    last_values = values
            else:
                # no overlap → save the previous one
                duration_data.append(last_values['duration'] / 13)
                energy_data.append(last_values['energy'])
                video_intervals.append(last_interval)
                video_scores.append(last_values['ratio'])
                video_areas.append(last_values['area'])

                # move to next candidate
                last_interval = (start, end)
                last_values = values

    # save the last candidate after the loop
    if last_interval is not None:
        duration_data.append(last_values['duration'] / 13)
        energy_data.append(last_values['energy'])
        video_intervals.append(last_interval)
        video_scores.append(last_values['ratio'])
        video_areas.append(last_values['area'])

    intervals_data.append(video_intervals)
    score_data.append(video_scores)
    area_data.append(video_areas)

mean1 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_0to3.npy')
mean2 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_4to7.npy')
mean3 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_8to11.npy')
mean4 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_12to15.npy')
mean5 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_16to19.npy')
mean6 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_20to23.npy')
mean7 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_24to27.npy')
mean8 = np.load(f'Fig 2 - statistics/MEAN_{video_name}_wf_raw_data_28to29.npy')

print(mean1.shape)
long_mean = np.concatenate([mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8])


data = [mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8]
num = 3

mean = data[num]
troughs = intervals_data[num]
long_mean = long_mean
long_troughs = list(chain.from_iterable(intervals_data))
scores_small = score_data[num]
scores = list(chain.from_iterable(score_data))
areas = area_data[num]
scale = 0.5
figsize_cm = (18, 9)
figsize_in = tuple(x / 2.54 for x in figsize_cm)


ticks_interval = 375  # 30 seconds = 375 frames

# ---- First Plot (ax1) ----
axes[0].plot(mean, label='Mean', color='black', linewidth=2*scale)
axes[0].set_xticks(np.arange(0, len(mean), ticks_interval))
axes[0].set_xticklabels([f'{np.round(i / 750,2) }' for i in np.arange(0, len(mean), ticks_interval)])

# Plot interval lines
for start, end in troughs:
    axes[0].plot([start%3700, end%3700], [1, 1], color = "firebrick", linewidth=4*scale, linestyle='-', zorder=10)
    axes[0].scatter([start%3700, end%3700], [1, 1], color='black', s=4, zorder=11)


# Add invisible plot for legend entry
axes[0].plot([], [], color="firebrick", linewidth=2*scale, linestyle='-', label='Interval')

specific_intervals = [(1521,1592, 'red')]

height = 1
for start, end, color in specific_intervals:
    rect = patches.Rectangle((start, 0), end - start, height, fill=False, edgecolor = 'slategrey' , linewidth=2*scale , zorder=10,linestyle ='--')
    axes[0].add_patch(rect)

# Scatter points at interval start & end
flattened_troughs = [point%3700 for tup in troughs for point in tup]  # Flatten tuple list
print([(troughs[i][0]%3700,troughs[i][1]%3700) for i in range (len(troughs))])

#for point in flattened_troughs:
#    axes[0].hlines(y=1, xmin=point - 3, xmax=point + 3, color='black', linewidth=20*scale, zorder=10)

#ax1.set_title('Data Events')
axes[0].set_ylim([-0.05, 1.1])
axes[0].set_ylabel('Avg Intesity',fontsize=10)
axes[0].set_yticks([0,0.5, 1])
axes[0].set_yticklabels(["0", "0.5", "1"])

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].axhline(0, color='black', zorder=5, linewidth=0.8*scale)
axes[0].tick_params(axis='x', labelsize=8)
axes[0].tick_params(axis='y', labelsize=8)
#axes[0].set_xlim([355, 1910])
#axes[0].set_xlim([1430, 2990])
axes[0].set_xlim([750*1+270, 750*3+270])
axes[0].set_title(f"{video_name}")

axes[0].set_xticks([])

axes[0].legend(
    frameon=False,  # removes the legend box
    loc='upper left',  # position (e.g., 'best', 'lower left', etc.)
    fontsize=8,  # font size
    ncol=1,  # number of columns
    handlelength=2  # length of the legend line
)

# ---- Second Plot (ax2) ----
time_stamps = [((start% 3700 + (end% 3700 - start% 3700) / 2) ) for start, end in troughs]

axes[1].set_ylim([-0.1, 1])
axes[1].set_ylabel('Waviness',color=score_color)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)
axes[1].spines['bottom'].set_position(('data', -0.05))  # move tick line to y=0


axes[1].set_xlabel('Time [min]',labelpad=0)
axes[1].set_xticks(np.arange(0, len(mean), ticks_interval))
axes[1].set_xticklabels([f'{np.round(i / 750,2) }' for i in np.arange(0, len(mean), ticks_interval)])

tick_positions = np.arange(750*1+270, 750*3+270 + 1, 750)  # 0, 1, 2, 3 minutes from 1325
tick_labels = ['0', '1', '2' ]
axes[1].set_xticks(tick_positions)
axes[1].set_xticklabels(tick_labels)


axes[1].axhline(0, color='black', zorder=5, linewidth=0.8)
axes[1].tick_params(axis='x', labelsize=8,direction='inout',length=4)
axes[1].set_yticks([0,0.5, 1])
axes[1].set_yticklabels(["0", "0.5", "1"])

axes[1].scatter(time_stamps, scores_small, color=score_color, label='Waviness', zorder=10, s = 35*scale , alpha = 1)
axes[1].plot(time_stamps, scores_small, color=score_color, label='Waviness', zorder=10 , alpha = 0.8 , linewidth = 2*scale)


#start_frame = 0
#end_frame = 3700
#filtered_time_stamps = [t for t in time_stamps if start_frame <= t <= end_frame]
#filtered_scores = [s for t, s in zip(time_stamps, scores_small) if start_frame <= t <= end_frame]
#axes[1].scatter(filtered_time_stamps, filtered_scores, color=score_color,label='Waviness', zorder=10, s=25*scale, alpha=0.8)
#axes[1].plot(filtered_time_stamps, filtered_scores, color=score_color,zorder=10, alpha=0.8, linewidth=2*scale)
axes[1].tick_params(axis='x', labelsize=8)
axes[1].tick_params(axis='y', labelsize=8)
axes[1].set_xlim([750*1+270, 750*3+270])

ax2 = axes[1].twinx()
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.scatter(time_stamps, areas, color=area_color, label='Active Area', zorder=10, alpha=1, s=35*scale )
ax2.set_yticks([0,0.5, 1])
ax2.set_yticklabels(["0", "0.5", "1"])
ax2.set_ylim([-0.1, 1])  # <--- This line makes sure ax2 y-limits match axes[1]

# Set y-axis label
ax2.set_ylabel('Active Area', color=area_color,fontsize = 10)

# Style the y-axis ticks and label in green
ax2.tick_params(axis='y', labelcolor='black', labelsize=8)



###### AX3 #####
ticks_interval = 750  # 1 min  = 780 frames
time_stamps = [(start + (end - start) / 2) for start, end in long_troughs]



axes[2].plot(time_stamps, scores, color=score_color, label='Waviness', zorder=10, linewidth=2*scale , alpha = 1)

axes[2].set_ylim([-0.01, 1])
axes[2].set_ylabel('Waviness')

axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].set_xlim([0,len(long_mean)])
axes[2].set_yticks([0,0.5, 1])
axes[2].set_yticklabels(["0", "0.5", "1"])

rect = plt.Rectangle((num*3700+520 ,0), 750*2, 0.3, color='gold', alpha=0.25, zorder = 0)
axes[2].add_patch(rect)

# ax2.set_xlim([0, 3640])
tick_interval_frames = 5 * 750  # 3900

tick_positions = np.arange(0, len(long_mean), tick_interval_frames)
tick_labels = [f'{i // 750}' for i in tick_positions]  # Convert frames to minutes

axes[2].set_xticks(tick_positions)
axes[2].set_xticklabels(tick_labels, zorder=1)
axes[2].set_xlabel('Time [min]', labelpad=-2)
axes[2].axhline(0, color='black', zorder=5, linewidth=0.8)
axes[2].tick_params(axis='x', labelsize=8)
axes[2].tick_params(axis='y', labelsize=8)
#ax3.scatter(time_stamps, areas, color='mediumseagreen', label='Score', zorder=10, linewidth=1*scale , alpha = 0.8)



############################



experiments = [
    '54MRL',
    '63MR',
    '187FN',
    '203MN',
    '204FR',
    '206FRL',
    '211MRR',
    '218MN',
    '21ML',
    '221ML'
]

all_data = {experiment: np.load(os.path.join( statistics_path, f'{experiment}_statistics.npy'), allow_pickle=True ).item() for experiment in experiments }


data_54MRL = all_data['54MRL']
data_63MR = all_data['63MR']
data_187FN = all_data['187FN']
data_203MN = all_data['203MN']
data_204FR = all_data['204FR']
data_206FRL = all_data['206FRL']
data_211MRR = all_data['211MRR']
data_218MN = all_data['218MN']
data_21ML = all_data['21ML']
data_221ML = all_data['221ML']



def process_data(data, min_area, offset):
    scores, areas, durations, energies, intervals = [], [], [], [], []

    for i, video in enumerate(data):
        last_end = -1  # To track the end of the last valid interval

        for interval, values in sorted(data[video].items()):
            start, end = interval[0] + offset * i, interval[1] + offset * i
            if values['area'] > min_area and start >= last_end:
                scores.append(values['ratio'])
                areas.append(values['area'])
                durations.append(values['duration'] / 13)
                energies.append(values['energy'])
                intervals.append((start, end))
                last_end = end  # Update the end of the last accepted interval
                #if values['ratio']>0.34:
                #    print(video ,'frame ',start%3700,'to',end%3700,"ratio: ",values['ratio'], "area: ",values['area'], "duration: ",np.round(values['duration'] / 13,3))

                if values['ratio'] > 0.42: # and values['area']>0.7:
                    print(video, i, start % 3700, end % 3700, "Ratio: ", values['ratio'], "Area: ",
                          values['area'])


    return scores, areas, durations, energies, intervals

offset = 3700

score_54MRL, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL, min_area, offset)
score_63MR, area_63MR, duration_63MR, energy_63MR, intervals_63MR = process_data(data_63MR, min_area, offset)
score_187FN, area_187FN, duration_187FN, energy_187FN, intervals_187FN = process_data(data_187FN, min_area, offset)
score_203MN, area_203MN, duration_203MN, energy_203MN, intervals_203MN = process_data(data_203MN, min_area, offset)
score_204FR, area_204FR, duration_204FR, energy_204FR, intervals_204FR = process_data(data_204FR, min_area, offset)
score_206FRL, area_206FRL, duration_206FRL, energy_206FRL, intervals_206FRL = process_data(data_206FRL, min_area, offset)
score_211MRR, area_211MRR, duration_211MRR, energy_211MRR, intervals_211MRR = process_data(data_211MRR, min_area, offset)
score_218MN, area_218MN, duration_218MN, energy_218MN, intervals_218MN = process_data(data_218MN, min_area, offset)
score_21ML, area_21ML, duration_21ML, energy_21ML, intervals_21ML = process_data(data_21ML, min_area, offset)
score_221ML, area_221ML, duration_221ML, energy_221ML, intervals_221ML = process_data(data_221ML, min_area, offset)




scores = [score_54MRL, score_63MR, score_187FN, score_203MN, score_204FR , score_206FRL , score_211MRR , score_218MN ,score_21ML,score_221ML]
areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML,area_221ML]
lengths = [duration_54MRL, duration_63MR, duration_187FN, duration_203MN, duration_204FR , duration_206FRL , duration_211MRR , duration_218MN , duration_21ML,duration_221ML]
energies = [energy_54MRL, energy_63MR, energy_187FN, energy_203MN, energy_204FR , energy_206FRL , energy_211MRR , energy_218MN , energy_21ML,energy_221ML]


total_values = sum(len(s) for s in scores)
print(total_values)
#print(total_values)
def two_D_histograms(ax, scores, areas , type,cbar):
    # Flatten the scores and areas if they are lists of lists or nested arrays
    try:
        scores = np.concatenate(scores)
    except Exception:
        scores = np.array(scores)

    try:
        areas = np.concatenate(areas)
    except Exception:
        areas = np.array(areas)

    white_to_green = mcolors.LinearSegmentedColormap.from_list(
        "white_to_green",
        [
            "#ffffff",
            area_color
        ],
        N=256
    )

    white_to_red = mcolors.LinearSegmentedColormap.from_list(
        "white_to_green",
        [
            "#ffffff",
            length_color
        ],
        N=256
    )

    # Create a custom colormap where 0 is white
    if type == 'lengths':
        cmap = white_to_red
    elif type == 'areas':
        cmap = white_to_green
    elif type == 'energies':
        cmap = plt.cm.Purples

    cmap = cmap(np.arange(cmap.N))  # Extract color array
    cmap[0, :] = [1, 1, 1, 1]  # Set the first color (0 value) to white
    custom_cmap = ListedColormap(cmap)

    # Compute 2D histogram


    # Plot the heatmap on the provided axis
    if type == 'lengths':
        hist, xedges, yedges = np.histogram2d(scores, areas, bins=[np.linspace(0, 13, 13), np.linspace(0, 1, 11)])

        # Normalize to percentage
        hist_percentage = (hist / hist.sum()) * 100  # Convert to percentage
        img = ax.pcolormesh(xedges, yedges, hist_percentage.T, cmap=custom_cmap, shading='auto', vmin=0, vmax=25)

    elif type == 'areas':
        hist, xedges, yedges = np.histogram2d(scores, areas, bins=[np.linspace(0, 1, 11), np.linspace(0, 1, 11)])

        # Normalize to percentage
        hist_percentage = (hist / hist.sum()) * 100  # Convert to percentage
        img = ax.pcolormesh(xedges, yedges, hist_percentage.T, cmap=custom_cmap, shading='auto', vmin=0, vmax=25)

    elif type == 'energies':
        hist, xedges, yedges = np.histogram2d(scores, areas, bins=[np.linspace(0, 1, 11), np.linspace(0, 1, 11)])

        # Normalize to percentage
        hist_percentage = (hist / hist.sum()) * 100  # Convert to percentage
        img = ax.pcolormesh(xedges, yedges, hist_percentage.T, cmap=custom_cmap, shading='auto', vmin=0, vmax=20)

    if cbar == True:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Create a horizontal colorbar below the axis without shrinking the axis itself
        cax = inset_axes(ax,
                         width="60%",  # width relative to parent axis
                         height="5%",  # height as percentage of parent
                         loc='lower center',
                         bbox_to_anchor=(0, 0.95, 1, 1.5),  # (x, y, width, height), y<0 places it below
                         bbox_transform=ax.transAxes,
                         borderpad=0)

        cbar = plt.colorbar(img, cax=cax, orientation='horizontal' ,ticks = [img.norm.vmin,img.norm.vmax])
        cbar.set_label('%',labelpad=-8.5)
        cbar.ax.tick_params(labelsize=8)

alpha = 1



score_data = {
    "54MRL": score_54MRL,
    "63MR": score_63MR,
    "187FN": score_187FN,
    "203MN": score_203MN,
    "204FR": score_204FR,
    "206FRL": score_206FRL,
    "211MRR": score_211MRR,
    "218MN": score_218MN,
    "21ML": score_21ML,
    "221ML": score_221ML
}

area_data = {
    "54MRL": area_54MRL,
    "63MR": area_63MR,
    "187FN": area_187FN,
    "203MN": area_203MN,
    "204FR": area_204FR,
    "206FRL": area_206FRL,
    "211MRR": area_211MRR,
    "218MN": area_218MN,
    "21ML": area_21ML,
    "221ML": area_221ML
}

duration_data = {
    "54MRL": duration_54MRL,
    "63MR": duration_63MR,
    "187FN": duration_187FN,
    "203MN": duration_203MN,
    "204FR": duration_204FR,
    "206FRL": duration_206FRL,
    "211MRR": duration_211MRR,
    "218MN": duration_218MN,
    "21ML": duration_21ML,
    "221ML": duration_221ML
}



counts, bins, patches = axes[3].hist(score_data[video_name], bins=np.linspace(0, 1, 11), color=score_color, alpha=alpha,rwidth=1,edgecolor=score_edge, linewidth=linewidth)

# Manually convert counts to percentages
percentage = (counts / len(score_data[video_name])) * 100

# Update the y-values of the histogram bars to be percentages
for count, patch in zip(percentage, patches):
    patch.set_height(count)

axes[3].set_xlim([0, 1])
axes[3].set_xticks([0,0.5, 1])
axes[3].set_xticklabels(["0", "0.5", "1"])

axes[3].set_ylim([0,100])
axes[3].set_ylabel('%',fontsize=10,labelpad=-0.7)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].tick_params(axis='x', labelsize=8)
axes[3].tick_params(axis='y', labelsize=8)


# Fifth plot (bottom-center) - Areas 54MRL
counts, bins, patches = axes[4].hist(area_data[video_name], bins=np.linspace(0, 1, 11), color=area_color, alpha=alpha, rwidth=1,edgecolor=area_edge, linewidth=linewidth)

# Manually convert counts to percentages
percentage = (counts / len(area_data[video_name])) * 100

# Update the y-values of the histogram bars to be percentages
for count, patch in zip(percentage, patches):
    patch.set_height(count)

axes[4].set_xlim([0, 1])
axes[4].set_xticks([0,0.5, 1])
axes[4].set_xticklabels(["0", "0.5", "1"])
axes[4].set_ylim([0,40])
axes[4].spines['top'].set_visible(False)
axes[4].spines['right'].set_visible(False)
axes[4].tick_params(axis='x', labelsize=8)
axes[4].tick_params(axis='y', labelsize=8)

# Fifth plot (bottom-center) - Durations 54MRL
counts, bins, patches = axes[5].hist(duration_data[video_name], bins=np.linspace(0, 15, 16), color=length_color, alpha=alpha, rwidth=1,edgecolor=length_edge,  linewidth=linewidth)

# Manually convert counts to percentages
percentage = (counts / len(duration_data[video_name])) * 100

# Update the y-values of the histogram bars to be percentages
for count, patch in zip(percentage, patches):
    patch.set_height(count)

#axes[5].set_xlabel('Length [sec]')
axes[5].set_xlim([0,15])
axes[5].set_xticks([0,5,10,15])
axes[5].set_xticklabels(["0", "5", "10","15"])
axes[5].set_ylim([0,30])
axes[5].set_yticks([0,10,20,30])
axes[5].set_yticklabels(["0", "10","20","30"])
axes[5].spines['top'].set_visible(False)
axes[5].spines['right'].set_visible(False)
axes[5].tick_params(axis='x', labelsize=8)
axes[5].tick_params(axis='y', labelsize=8)


two_D_histograms(axes[6], area_data[video_name], score_data[video_name] , type = 'areas',cbar = True)
axes[6].set_xlim([0,1])
axes[6].set_xticks([0,0.5, 1])
axes[6].set_xticklabels(["0", "0.5", "1"])

axes[6].set_ylim([0,1])
axes[6].set_yticks([0, 1])
axes[6].set_ylabel("Score",fontsize=10,labelpad=-5)
axes[6].tick_params(axis='x', labelsize=8)
axes[6].tick_params(axis='y', labelsize=8)
axes[6].spines['top'].set_visible(False)
axes[6].spines['right'].set_visible(False)


two_D_histograms(axes[7], duration_data[video_name], score_data[video_name] , type = 'lengths',cbar = True)
axes[7].set_xticks([0,5,10,15])
axes[7].set_xlim([0,15])
axes[7].set_xticklabels(["0", "5", "10","15"])
axes[7].set_ylim([0,1])
axes[7].set_yticks([0, 1])
#axes[7].set_ylabel("Length [sec]",fontsize=10)
axes[7].tick_params(axis='x', labelsize=8)
axes[7].tick_params(axis='y', labelsize=8)
axes[7].spines['top'].set_visible(False)
axes[7].spines['right'].set_visible(False)




# Define datasets for the first 3 plots
all_datasets = [scores, areas, lengths]
titles = ['Waviness', 'Active Area', 'Lengths']
colors = [score_color, area_color, length_color]
edge_colors = [score_edge, area_edge, length_edge]


bin_list = [
    np.histogram_bin_edges(np.concatenate(scores), bins=np.linspace(0, 1, 11)),
    np.histogram_bin_edges(np.concatenate(areas), bins=10),
    np.histogram_bin_edges(np.concatenate(lengths), bins=20)
]


# Plot the first 3 figures (as in your loop)
for i, (dataset, ax, title, color, bins,edge_colors) in enumerate(zip(all_datasets, [axes[8],axes[9],axes[10]], titles, colors, bin_list,edge_colors)):
    histograms = [np.histogram(data, bins=bins)[0] for data in dataset]
    hist_array = np.array(histograms)

    total_counts = np.sum(hist_array, axis=1, keepdims=True)
    prob_array = hist_array / total_counts

    mean_probs = np.mean(prob_array, axis=0)
    sem_probs = np.std(prob_array, axis=0, ddof=1) / np.sqrt(len(dataset))

    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.bar(bin_centers, mean_probs * 100, width=np.diff(bins), color=color, alpha=alpha,label="Mean Probability",edgecolor=edge_colors,linewidth=linewidth,antialiased=True, align='center' )
    ax.errorbar(bin_centers, mean_probs * 100, yerr=sem_probs * 100, fmt='.', color='black',markersize=5, capsize=2,label="SEM Error")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    #ax.set_title(title)
    ax.set_ylabel('%' if i == 0 else '',labelpad=-0.7)
    ax.grid(False)

# Set specific axis limits for first row
axes[8].set_xlim([0, 1])
axes[8].set_ylim([0, 100])
axes[8].set_xlabel('Waviness',fontsize=10)
axes[8].set_xticks([0,0.5, 1])
axes[8].set_xticklabels(["0", "0.5", "1"])


axes[9].set_ylim([0, 40])
axes[9].set_xlim([0, 1])
axes[9].set_xlabel('Active Area',fontsize=10)
axes[9].set_xticks([0,0.5, 1])
axes[9].set_xticklabels(["0", "0.5", "1"])


axes[10].set_ylim([0, 30])
axes[10].set_xlim([0,15])
axes[10].set_xlabel('Length [sec]',fontsize=10)
axes[10].set_yticks([0,10,20,30])
axes[10].set_yticklabels(["0", "10","20","30"])
axes[10].set_xticks([0,5,10,15])
#axes[10].set_xticklabels(["0", "5", "10","15"])



two_D_histograms(axes[11],areas, scores, type = 'areas',cbar = False)
axes[11].set_xlim([0,1])
axes[11].set_xticks([0,0.5, 1])
axes[11].set_xticklabels(["0", "0.5", "1"])
axes[11].set_xlabel("Active Area",fontsize=10)

axes[11].set_ylim([0,1])
axes[11].set_yticks([0, 1])
axes[11].set_ylabel("Waviness",fontsize=10,labelpad=-5)
axes[11].tick_params(axis='x', labelsize=8)
axes[11].tick_params(axis='y', labelsize=8)
axes[11].spines['top'].set_visible(False)
axes[11].spines['right'].set_visible(False)


two_D_histograms(axes[12], lengths, scores , type = 'lengths',cbar = False)
axes[12].set_xlim([0,15])
axes[12].set_xticks([0,5,10,15])
axes[12].set_xticklabels(["0", "5", "10","15"])
axes[12].set_ylim([0,1])
axes[12].set_yticks([0, 1])
axes[12].set_xlabel("Length [sec]",fontsize=10)
#axes[12].set_ylabel("Scores",fontsize=10)
axes[12].tick_params(axis='x', labelsize=8)
axes[12].tick_params(axis='y', labelsize=8)
axes[12].spines['top'].set_visible(False)
axes[12].spines['right'].set_visible(False)



for i in range(3,13):
    axes[i].set_aspect('equal', adjustable='box')
    x_min, x_max = axes[i].get_xlim()
    y_min, y_max = axes[i].get_ylim()
    axes[i].set_aspect((x_max - x_min) / (y_max - y_min), adjustable='box')






# Layout adjustments and legend
fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.07, wspace=0, hspace=0.5)

#plt.savefig("Video_mean__Score_mean__All_video_mean.svg")
plt.savefig("Figure 1.pdf" ,dpi=1000)

plt.show()





