import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import itertools
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


# Load data


plt.rcParams['svg.fonttype'] = 'none'  # Ensures text remains text



data_54MRL_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_54MRL_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_54MRL_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/54MRL/54MRL_all_data.npy', allow_pickle=True).item()
data_54MRL_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_54MRL_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_54MRL_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=1.npy', allow_pickle=True).item()
data_54MRL_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=2.npy', allow_pickle=True).item()
data_54MRL_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/54MRL_all_data_alpha=5.npy', allow_pickle=True).item()


data_218MN_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_218MN_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_218MN_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/218MN/218MN_all_data.npy', allow_pickle=True).item()
data_218MN_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_218MN_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_218MN_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=1.npy', allow_pickle=True).item()
data_218MN_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=2.npy', allow_pickle=True).item()
data_218MN_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/218MN_all_data_alpha=5.npy', allow_pickle=True).item()


data_206FRL_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_206FRL_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_206FRL_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/206FRL/206FRL_all_data.npy', allow_pickle=True).item()
data_206FRL_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_206FRL_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_206FRL_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=1.npy', allow_pickle=True).item()
data_206FRL_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=2.npy', allow_pickle=True).item()
data_206FRL_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/206FRL_all_data_alpha=5.npy', allow_pickle=True).item()


data_63MR_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_63MR_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_63MR_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/63MR/63MR_all_data.npy', allow_pickle=True).item()
data_63MR_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_63MR_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_63MR_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=1.npy', allow_pickle=True).item()
data_63MR_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=2.npy', allow_pickle=True).item()
data_63MR_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/63MR_all_data_alpha=5.npy', allow_pickle=True).item()


data_187FN_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_187FN_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_187FN_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/187FN/187FN_all_data.npy', allow_pickle=True).item()
data_187FN_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_187FN_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_187FN_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=1.npy', allow_pickle=True).item()
data_187FN_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=2.npy', allow_pickle=True).item()
data_187FN_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/187FN_all_data_alpha=5.npy', allow_pickle=True).item()


data_203MN_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_203MN_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_203MN_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/203MN/203MN_all_data.npy', allow_pickle=True).item()
data_203MN_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_203MN_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_203MN_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=1.npy', allow_pickle=True).item()
data_203MN_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=2.npy', allow_pickle=True).item()
data_203MN_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/203MN_all_data_alpha=5.npy', allow_pickle=True).item()

data_204FR_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_204FR_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_204FR_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/204FR/204FR_all_data.npy', allow_pickle=True).item()
data_204FR_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_204FR_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_204FR_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=1.npy', allow_pickle=True).item()
data_204FR_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=2.npy', allow_pickle=True).item()
data_204FR_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/204FR_all_data_alpha=5.npy', allow_pickle=True).item()


data_211MRR_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_211MRR_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_211MRR_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/211MRR/211MRR_all_data.npy', allow_pickle=True).item()
data_211MRR_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_211MRR_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_211MRR_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=1.npy', allow_pickle=True).item()
data_211MRR_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=2.npy', allow_pickle=True).item()
data_211MRR_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/211MRR_all_data_alpha=5.npy', allow_pickle=True).item()


data_21ML_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_21ML_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_21ML_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/21ML/21ML_all_data.npy', allow_pickle=True).item()
data_21ML_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_21ML_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_21ML_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=1.npy', allow_pickle=True).item()
data_21ML_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=2.npy', allow_pickle=True).item()
data_21ML_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/21ML_all_data_alpha=5.npy', allow_pickle=True).item()

data_221ML_005 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=0.05.npy', allow_pickle=True).item()
data_221ML_01 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=0.1.npy', allow_pickle=True).item()
data_221ML_03 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/221ML/221ML_all_data.npy', allow_pickle=True).item()
data_221ML_05 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=0.5.npy', allow_pickle=True).item()
data_221ML_07 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=0.7.npy', allow_pickle=True).item()
data_221ML_1 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=1.npy', allow_pickle=True).item()
data_221ML_2 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=2.npy', allow_pickle=True).item()
data_221ML_5 = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/diff alpha/221ML_all_data_alpha=5.npy', allow_pickle=True).item()



def activity_map():
    activity_map_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/54MRL/54MRL_activity_map.npy')
    activity_map_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/63MR/63MR_activity_map.npy')
    activity_map_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/187FN/187FN_activity_map.npy')
    activity_map_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/203MN/203MN_activity_map.npy')
    activity_map_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/204FR/204FR_activity_map.npy')
    activity_map_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/206FRL/206FRL_activity_map.npy')
    activity_map_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/211MRR/211MRR_activity_map.npy')
    activity_map_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/218MN/218MN_activity_map.npy')
    activity_map_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/21ML/21ML_activity_map.npy')

    activity_map = activity_map_54MRL + activity_map_63MR + activity_map_187FN + activity_map_203MN + activity_map_204FR + activity_map_206FRL + activity_map_211MRR + activity_map_218MN + activity_map_21ML


    plt.imshow(activity_map,  cmap='tab20c')  # Use 'gray' colormap for grayscale
    plt.colorbar()  # Add color bar for better understanding
    plt.title('128x128 Array Visualization')
    plt.axis('off')  # Hide axes for a cleaner view (optional)
    plt.show()



min_area=-0.1


def process_data(data, min_area, offset,fps=13):
    scores, areas, durations, energies, intervals = [], [], [], [], []

    for i, video in enumerate(data):
        last_end = -1  # To track the end of the last valid interval

        for interval, values in sorted(data[video].items()):
            start, end = interval[0] + offset * i, interval[1] + offset * i
            if values['area'] > min_area and start >= last_end and values['duration']/fps<30:
                scores.append(values['ratio'])
                areas.append(values['area'])
                durations.append(values['duration'] / fps)
                energies.append(values['energy'])
                intervals.append((start, end))
                last_end = end  # Update the end of the last accepted interval
                #if values['ratio'] > 0.13:
                #    print(video, (start %3700, end %3700),values['ratio'], values['area'] )
                #if values['area'] > 0.9:
                #    print(video, (start %3700, end %3700), values['area'],values['ratio'] )

    return scores, areas, durations, energies, intervals


offset = 3700


score_54MRL_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_005, min_area, offset)
score_54MRL_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_01, min_area, offset)
score_54MRL_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_03, min_area, offset)
score_54MRL_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_05, min_area, offset)
score_54MRL_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_07, min_area, offset)
score_54MRL_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_1, min_area, offset)
score_54MRL_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_2, min_area, offset)
score_54MRL_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL_5, min_area, offset)

score_218MN_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_005, min_area, offset)
score_218MN_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_01, min_area, offset)
score_218MN_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_03, min_area, offset)
score_218MN_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_05, min_area, offset)
score_218MN_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_07, min_area, offset)
score_218MN_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_1, min_area, offset)
score_218MN_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_2, min_area, offset)
score_218MN_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_218MN_5, min_area, offset)


score_206FRL_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_005, min_area, offset)
score_206FRL_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_01, min_area, offset)
score_206FRL_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_03, min_area, offset)
score_206FRL_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_05, min_area, offset)
score_206FRL_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_07, min_area, offset)
score_206FRL_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_1, min_area, offset)
score_206FRL_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_2, min_area, offset)
score_206FRL_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_206FRL_5, min_area, offset)


score_63MR_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_005, min_area, offset)
score_63MR_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_01, min_area, offset)
score_63MR_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_03, min_area, offset)
score_63MR_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_05, min_area, offset)
score_63MR_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_07, min_area, offset)
score_63MR_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_1, min_area, offset)
score_63MR_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_2, min_area, offset)
score_63MR_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_63MR_5, min_area, offset)


score_187FN_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_005, min_area, offset)
score_187FN_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_01, min_area, offset)
score_187FN_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_03, min_area, offset)
score_187FN_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_05, min_area, offset)
score_187FN_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_07, min_area, offset)
score_187FN_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_1, min_area, offset)
score_187FN_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_2, min_area, offset)
score_187FN_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_5, min_area, offset)


score_203MN_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_005, min_area, offset)
score_203MN_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_01, min_area, offset)
score_203MN_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_03, min_area, offset)
score_203MN_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_05, min_area, offset)
score_203MN_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_187FN_07, min_area, offset)
score_203MN_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_203MN_1, min_area, offset)
score_203MN_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_203MN_2, min_area, offset)
score_203MN_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_203MN_5, min_area, offset)


score_204FR_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_005, min_area, offset)
score_204FR_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_01, min_area, offset)
score_204FR_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_03, min_area, offset)
score_204FR_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_05, min_area, offset)
score_204FR_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_07, min_area, offset)
score_204FR_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_1, min_area, offset)
score_204FR_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_2, min_area, offset)
score_204FR_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_204FR_5, min_area, offset)


score_211MRR_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_005, min_area, offset)
score_211MRR_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_01, min_area, offset)
score_211MRR_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_03, min_area, offset)
score_211MRR_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_05, min_area, offset)
score_211MRR_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_07, min_area, offset)
score_211MRR_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_1, min_area, offset)
score_211MRR_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_2, min_area, offset)
score_211MRR_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_211MRR_5, min_area, offset)


score_21ML_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_005, min_area, offset)
score_21ML_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_01, min_area, offset)
score_21ML_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_03, min_area, offset)
score_21ML_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_05, min_area, offset)
score_21ML_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_07, min_area, offset)
score_21ML_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_1, min_area, offset)
score_21ML_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_2, min_area, offset)
score_21ML_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_21ML_5, min_area, offset)


score_221ML_005, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_005, min_area, offset)
score_221ML_01, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_01, min_area, offset)
score_221ML_03, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_03, min_area, offset)
score_221ML_05, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_05, min_area, offset)
score_221ML_07, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_07, min_area, offset)
score_221ML_1, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_1, min_area, offset)
score_221ML_2, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_2, min_area, offset)
score_221ML_5, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_221ML_5, min_area, offset)





scores_by_alpha = {
    0.05: score_54MRL_005 + score_218MN_005 + score_206FRL_005 + score_63MR_005 + score_187FN_005 + score_203MN_005 + score_204FR_005 + score_211MRR_005 + score_21ML_005  + score_221ML_005,
    0.1:  score_54MRL_01  + score_218MN_01  + score_206FRL_01  + score_63MR_01  + score_187FN_01  + score_203MN_01  + score_204FR_01  + score_211MRR_01  + score_21ML_01   + score_221ML_01,
    0.3:  score_54MRL_03  + score_218MN_03  + score_206FRL_03  + score_63MR_03  + score_187FN_03  + score_203MN_03  + score_204FR_03  + score_211MRR_03  + score_21ML_03   + score_221ML_03,
    0.5:  score_54MRL_05  + score_218MN_05  + score_206FRL_05  + score_63MR_05  + score_187FN_05  + score_203MN_05  + score_204FR_05  + score_211MRR_05  + score_21ML_05   + score_221ML_05,
    0.7:  score_54MRL_07  + score_218MN_07  + score_206FRL_07  + score_63MR_07  + score_187FN_07  + score_203MN_07  + score_204FR_07  + score_211MRR_07  + score_21ML_07   + score_221ML_07,
    1:    score_54MRL_1   + score_218MN_1   + score_206FRL_1   + score_63MR_1   + score_187FN_1   + score_203MN_1   + score_204FR_1   + score_211MRR_1   + score_21ML_1    + score_221ML_1,
    2:    score_54MRL_2   + score_218MN_2   + score_206FRL_2   + score_63MR_2   + score_187FN_2   + score_203MN_2   + score_204FR_2   + score_211MRR_2   + score_21ML_2    + score_221ML_2,
    5:    score_54MRL_5   + score_218MN_5   + score_206FRL_5   + score_63MR_5   + score_187FN_5   + score_203MN_5   + score_204FR_5   + score_211MRR_5   + score_21ML_5    + score_221ML_5
}

# Ensure sorted alphas
alphas = sorted(scores_by_alpha.keys())
data = [scores_by_alpha[a] for a in alphas]

# Compute mean & median
means = [np.mean(d) for d in data]
medians = [np.median(d) for d in data]
print(means)
figsize_cm = (11, 4.8)  # example in cm
figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches

plt.figure(figsize=figsize_in)

# Boxplot (robust distribution)
plt.boxplot(
    data,
    positions=range(len(alphas)),
    widths=0.7,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor='lightgray', edgecolor='black'),
    medianprops=dict(color='black', linewidth=2)
)

# Plot markers individually
for i, y in enumerate(means):
    size = 20   # smaller marker for first point
    if i==0: plt.scatter(i, y, color='royalblue', s=size,zorder=2,label='Mean score')
    else: plt.scatter(i, y, color='royalblue', s=size,zorder=2)

# Axis formatting
plt.xticks(range(len(alphas)), [str(a) for a in alphas])
plt.xlabel(r"Regularization parameter $\alpha$")
plt.ylabel("Waviness score")
plt.ylim([0,1])
plt.yticks([0,0.25,0.5,0.75,1])


plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("alpha_statistics_summary.pdf", dpi=300)
plt.show()
