import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib
from scipy.stats import gaussian_kde
import itertools
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42

def sigma_0_02() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.02/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))

def sigma_0_03() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.03/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))

def sigma_0_04() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.04/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))

def sigma_0_05() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.05/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))

def sigma_0_06() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.06/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))

def sigma_0_08() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.08/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))

def sigma_0_1() :
    data_54MRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/54MRL_all_data.npy', allow_pickle=True).item()
    data_63MR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/63MR_all_data.npy',allow_pickle=True).item()
    data_187FN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/187FN_all_data.npy', allow_pickle=True).item()
    data_203MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/203MN_all_data.npy', allow_pickle=True).item()
    data_204FR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/204FR_all_data.npy', allow_pickle=True).item()
    data_206FRL = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/206FRL_all_data.npy', allow_pickle=True).item()
    data_211MRR = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/211MRR_all_data.npy', allow_pickle=True).item()
    data_218MN = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/218MN_all_data.npy', allow_pickle=True).item()
    data_21ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/21ML_all_data.npy', allow_pickle=True).item()
    data_221ML = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/beta=0.1/221ML_all_data.npy', allow_pickle=True).item()


    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL = [], [], [], []
    for video in data_54MRL.keys():
        for interval in data_54MRL[video].keys():
            score_54MRL.append(data_54MRL[video][interval]['ratio'])
            area_54MRL.append(data_54MRL[video][interval]['area'])
            duration_54MRL.append(data_54MRL[video][interval]['duration'] / 13)
            energy_54MRL.append(data_54MRL[video][interval]['energy'])

    score_63MR, area_63MR, duration_63MR, energy_63MR = [], [], [], []
    for video in data_63MR.keys():
        for interval in data_63MR[video].keys():
            score_63MR.append(data_63MR[video][interval]['ratio'])
            area_63MR.append(data_63MR[video][interval]['area'])
            duration_63MR.append(data_63MR[video][interval]['duration'] / 13)
            energy_63MR.append(data_63MR[video][interval]['energy'])

            #print(video,interval,data_63MR[video][interval]['ratio'] , data_63MR[video][interval]['area'] , np.round(data_63MR[video][interval]['duration']/13,2))

    score_187FN, area_187FN, duration_187FN, energy_187FN  = [], [], [], []
    for video in data_187FN.keys():
        for interval in data_187FN[video].keys():
            score_187FN.append(data_187FN[video][interval]['ratio'])
            area_187FN.append(data_187FN[video][interval]['area'])
            duration_187FN.append(data_187FN[video][interval]['duration'] / 13)
            energy_187FN.append(data_187FN[video][interval]['energy'])

    score_203MN, area_203MN, duration_203MN, energy_203MN  = [], [], [], []
    for video in data_203MN.keys():
        for interval in data_203MN[video].keys():
            score_203MN.append(data_203MN[video][interval]['ratio'])
            area_203MN.append(data_203MN[video][interval]['area'])
            duration_203MN.append(data_203MN[video][interval]['duration'] / 13)
            energy_203MN.append(data_203MN[video][interval]['energy'])

    score_204FR, area_204FR, duration_204FR, energy_204FR  = [], [], [], []
    for video in data_204FR.keys():
        for interval in data_204FR[video].keys():
            score_204FR.append(data_204FR[video][interval]['ratio'])
            area_204FR.append(data_204FR[video][interval]['area'])
            duration_204FR.append(data_204FR[video][interval]['duration'] / 13)
            energy_204FR.append(data_204FR[video][interval]['energy'])

    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL  = [], [], [], []
    for video in data_206FRL.keys():
        for interval in data_206FRL[video].keys():
            score_206FRL.append(data_206FRL[video][interval]['ratio'])
            area_206FRL.append(data_206FRL[video][interval]['area'])
            duration_206FRL.append(data_206FRL[video][interval]['duration'] / 13)
            energy_206FRL.append(data_206FRL[video][interval]['energy'])

    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR  = [], [], [], []
    for video in data_211MRR.keys():
        for interval in data_211MRR[video].keys():
            score_211MRR.append(data_211MRR[video][interval]['ratio'])
            area_211MRR.append(data_211MRR[video][interval]['area'])
            duration_211MRR.append(data_211MRR[video][interval]['duration'] / 13)
            energy_211MRR.append(data_211MRR[video][interval]['energy'])

    score_218MN, area_218MN, duration_218MN, energy_218MN  = [], [], [], []
    for video in data_218MN.keys():
        for interval in data_218MN[video].keys():
            score_218MN.append(data_218MN[video][interval]['ratio'])
            area_218MN.append(data_218MN[video][interval]['area'])
            duration_218MN.append(data_218MN[video][interval]['duration'] / 13)
            energy_218MN.append(data_218MN[video][interval]['energy'])

    score_21ML, area_21ML, duration_21ML, energy_21ML  = [], [], [], []
    for video in data_21ML.keys():
        for interval in data_21ML[video].keys():
            score_21ML.append(data_21ML[video][interval]['ratio'])
            area_21ML.append(data_21ML[video][interval]['area'])
            duration_21ML.append(data_21ML[video][interval]['duration'] / 13)
            energy_21ML.append(data_21ML[video][interval]['energy'])

    score_221ML, area_221ML, duration_221ML, energy_221ML  = [], [], [], []
    for video in data_221ML.keys():
        for interval in data_221ML[video].keys():
            score_221ML.append(data_221ML[video][interval]['ratio'])
            area_221ML.append(data_221ML[video][interval]['area'])
            duration_221ML.append(data_221ML[video][interval]['duration'] / 13)
            energy_221ML.append(data_221ML[video][interval]['energy'])

    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR , area_206FRL , area_211MRR , area_218MN , area_21ML, area_221ML]

    return list(itertools.chain(*areas))



areas_sigma_0_02 = sigma_0_02()
areas_sigma_0_03 = sigma_0_03()
areas_sigma_0_04 = sigma_0_04()
areas_sigma_0_05 = sigma_0_05()
areas_sigma_0_06 = sigma_0_06()
areas_sigma_0_08 = sigma_0_08()
areas_sigma_0_1 = sigma_0_1()





def two_D_histograms(ax, scores, areas , type):
    # Flatten the scores and areas if they are lists of lists or nested arrays
    scores = np.concatenate(scores) if isinstance(scores, list) else scores
    areas = np.concatenate(areas) if isinstance(areas, list) else areas

    # Create a custom colormap where 0 is white
    if type == 'lengths':
        cmap = plt.cm.Oranges
    elif type == 'areas':
        cmap = plt.cm.Greens
    elif type == 'energies':
        cmap = plt.cm.Purples

    cmap = cmap(np.arange(cmap.N))  # Extract color array
    cmap[0, :] = [1, 1, 1, 1]  # Set the first color (0 value) to white
    custom_cmap = ListedColormap(cmap)

    # Compute 2D histogram
    hist, xedges, yedges = np.histogram2d(scores, areas, bins=10)

    # Normalize to percentage
    hist_percentage = (hist / hist.sum()) * 100  # Convert to percentage

    # Plot the heatmap on the provided axis
    if type == 'lengths':
        img = ax.pcolormesh(xedges, yedges, hist_percentage.T, cmap=custom_cmap, shading='auto', vmin=0, vmax=35)
    elif type == 'areas':
        img = ax.pcolormesh(xedges, yedges, hist_percentage.T, cmap=custom_cmap, shading='auto', vmin=0, vmax=12)
    elif type == 'energies':
        img = ax.pcolormesh(xedges, yedges, hist_percentage.T, cmap=custom_cmap, shading='auto', vmin=0, vmax=20)

    ax.set_xlabel('Scores')
    #ax.set_ylabel("Length [sec]")
    #ax.set_title('Score VS Lengths of Events (Percentage)')

    # Add a colorbar
    cbar = plt.colorbar(img, ax=ax, label="Percentage (%)")
    cbar.ax.tick_params(labelsize=8)


def score_area(scores , areas , lengths ,alpha ,percent):
    fig, axes = plt.subplots(4, 3 , figsize=(10, 10))  # Adjust height to fit extra row

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    row_labels = [r' ', r'$\alpha=0.5$', r'$\alpha=1$', r'$\alpha=1.5$']
    col_labels = [r'$\beta=0.02$', r'$\beta=0.03$', r'$\beta=0.04$']

    fig.subplots_adjust(left=0.15, top=0.88)

    # Column headers
    for col in range(3):
        fig.text(0.25 + col * 0.25, 0.92, col_labels[col], ha='center', fontsize=14, weight='bold')

    # Row headers
    for row in range(4):
        fig.text(0.05, 0.78 - row * 0.2, row_labels[row], va='center', rotation='vertical', fontsize=14, weight='bold')



    ## Beta = 0.2 AREA
    counts, bins, patches = axes[0].hist(area_a1_b02_iter150, bins=10, color='mediumseagreen', alpha=alpha, rwidth=0.9)
    percentage = (counts / len(area_a1_b02_iter150)) * 100

    for count, patch in zip(percentage, patches):     # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    #axes[0].set_xlabel(r'Relative Area, $\beta$ = 0.02')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_ylim([0,40])





    ## Beta = 0.3 AREA
    counts, bins, patches = axes[1].hist(area_a1_b03_iter150, bins=10, color='mediumseagreen', alpha=alpha, rwidth=0.9)
    percentage = (counts / len(area_a1_b03_iter150)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    #axes[1].set_xlabel(r'Relative Area, $\beta$ = 0.03')
    axes[1].set_ylim([0,40])



    ## Beta = 0.4 AREA
    counts, bins, patches = axes[2].hist(area_a1_b04_iter150, bins=10, color='mediumseagreen', alpha=alpha, rwidth=0.9)
    percentage = (counts / len(area_a1_b04_iter150)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    #axes[2].set_xlabel(r'Relative Area, $\beta$ = 0.04')
    axes[2].set_ylim([0,40])



    #### alpha = 0.5 , beta = 0.2
    counts, bins, patches = axes[3].hist(score_a05_b02_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a05_b02_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[3].set_xlabel(r'Score $\alpha$ = 0.5, $\beta$ = 0.02')
    axes[3].set_ylabel('Percentage (%)')
    axes[3].set_xlim([0, 1])
    axes[3].set_ylim([0,100])


    #### alpha = 0.5 , beta = 0.3
    counts, bins, patches = axes[4].hist(score_a05_b03_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a05_b03_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[4].set_xlabel(r'Score $\alpha$ = 0.5, $\beta$ = 0.03')
    axes[4].set_xlim([0, 1])
    axes[4].set_ylim([0,100])


    #### alpha = 0.5 , beta = 0.4
    counts, bins, patches = axes[5].hist(score_a05_b04_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a05_b04_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[5].set_xlabel(r'Score $\alpha$ = 0.5, $\beta$ = 0.04')
    axes[5].set_xlim([0, 1])
    axes[5].set_ylim([0,100])




    #### alpha = 1 , beta = 0.2
    counts, bins, patches = axes[6].hist(score_a1_b02_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a1_b02_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[6].set_xlabel(r'Score $\alpha$ = 1 , $\beta$ =0.02')
    axes[6].set_ylabel('Percentage (%)')
    axes[6].set_xlim([0, 1])
    axes[6].set_ylim([0,100])



    #### alpha = 1 , beta = 0.3
    counts, bins, patches = axes[7].hist(score_a1_b03_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a1_b03_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[7].set_xlabel(r'Score $\alpha$ = 1, $\beta$ = 0.03')
    axes[7].set_xlim([0, 1])
    axes[7].set_ylim([0,100])



    #### alpha = 1 , beta = 0.4

    counts, bins, patches = axes[8].hist(score_a1_b04_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a1_b04_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[8].set_xlabel(r'Score $\alpha$ = 1, $\beta$ = 0.04')
    axes[8].set_xlim([0, 1])
    axes[8].set_ylim([0,100])





    #### alpha = 1.5 , beta = 0.2

    counts, bins, patches = axes[9].hist(score_a15_b02_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a15_b02_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[9].set_xlabel(r'Score $\alpha$ = 1.5, $\beta$ = 0.02')
    axes[9].set_ylabel('Percentage (%)')
    axes[9].set_xlim([0, 1])
    axes[9].set_ylim([0,100])

    #### alpha = 1.5 , beta = 0.3

    counts, bins, patches = axes[10].hist(score_a15_b03_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a15_b03_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[10].set_xlabel(r'Score $\alpha$ = 1.5, $\beta$ = 0.03')
    axes[10].set_xlim([0, 1])
    axes[10].set_ylim([0,100])


    #### alpha = 1.5 , beta = 0.4

    counts, bins, patches = axes[11].hist(score_a15_b04_iter150, bins=np.linspace(0, 1, 21), color='royalblue', alpha=alpha,rwidth=0.9)
    percentage = (counts / len(score_a15_b04_iter150)) * 100

    for count, patch in zip(percentage, patches):
        patch.set_height(count)

    #axes[11].set_xlabel(r'Score $\alpha$ = 1.5, $\beta$ = 0.04')
    axes[11].set_xlim([0, 1])
    axes[11].set_ylim([0,100])




    plt.tight_layout()


    #plt.savefig("All_videos_extended.svg", format="svg", bbox_inches="tight", dpi=300, transparent=True)
    plt.show()

def area(alpha):
    figsize_cm = (18, 4.5)  # example in cm
    figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches

    fig, axes = plt.subplots(1, 7 , figsize=figsize_in)  # Adjust height to fit extra row

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    ## Beta = 0.02 AREA
    counts, bins, patches = axes[0].hist(areas_sigma_0_02, bins=10, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_02)) * 100

    for count, patch in zip(percentage, patches):     # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    fontsize = 10
    axes[0].set_title(r'$\sigma$ = 0.02',fontsize=fontsize)
    axes[0].set_xlabel('Relative Area',fontsize=fontsize)
    axes[0].set_ylabel('Events [%]',fontsize=fontsize)
    axes[0].set_ylim([0,60])
    axes[0].set_xlim([0,1])



    ## Beta = 0.02 AREA
    counts, bins, patches = axes[1].hist(areas_sigma_0_03, bins=10, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_03)) * 100

    for count, patch in zip(percentage, patches):     # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    axes[1].set_title(r'$\sigma$ = 0.03',fontsize=fontsize)
    axes[1].set_ylim([0,60])
    axes[1].set_xlim([0,1])



    ## Beta = 0.04 AREA
    counts, bins, patches = axes[2].hist(areas_sigma_0_04, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_04)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    axes[2].set_title(r'$\sigma$ = 0.04',fontsize=fontsize)
    axes[2].set_ylim([0,60])
    axes[2].set_xlim([0,1])


    ## Beta = 0.05 AREA
    counts, bins, patches = axes[3].hist(areas_sigma_0_05, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_06)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    axes[3].set_title(r'$\sigma$ = 0.05',fontsize=fontsize)
    axes[3].set_ylim([0,60])
    axes[3].set_xlim([0,1])

    ## Beta = 0.06 AREA
    counts, bins, patches = axes[4].hist(areas_sigma_0_06, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_06)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    axes[4].set_title(r'$\sigma$ = 0.06',fontsize=fontsize)
    axes[4].set_ylim([0,60])
    axes[4].set_xlim([0,1])


    ## Beta = 0.08 AREA
    counts, bins, patches = axes[5].hist(areas_sigma_0_08, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_08)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    axes[5].set_title(r'$\sigma$ = 0.08',fontsize=fontsize)
    axes[5].set_ylim([0,60])
    axes[5].set_xlim([0,1])

    ## Beta = 0.1 AREA
    counts, bins, patches = axes[6].hist(areas_sigma_0_1, color='#6baa91', alpha=alpha, rwidth=1,edgecolor='#000000',linewidth=0.5)
    percentage = (counts / len(areas_sigma_0_1)) * 100

    for count, patch in zip(percentage, patches):    # Update the y-values of the histogram bars to be percentages
        patch.set_height(count)

    axes[6].set_title(r'$\sigma$ = 0.1',fontsize=fontsize)
    axes[6].set_ylim([0,60])
    axes[6].set_xlim([0,1])


    axes[0].tick_params(axis='x', labelsize=8)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[2].tick_params(axis='x', labelsize=8)
    axes[3].tick_params(axis='x', labelsize=8)
    axes[4].tick_params(axis='x', labelsize=8)
    axes[5].tick_params(axis='x', labelsize=8)
    axes[6].tick_params(axis='x', labelsize=8)

    axes[0].set_yticks([0,30,60])
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    axes[3].set_yticks([])
    axes[4].set_yticks([])
    axes[5].set_yticks([])
    axes[6].set_yticks([])






    axes[0].set_box_aspect(1)
    axes[1].set_box_aspect(1)
    axes[2].set_box_aspect(1)
    axes[3].set_box_aspect(1)
    axes[4].set_box_aspect(1)
    axes[5].set_box_aspect(1)
    axes[6].set_box_aspect(1)

    plt.tight_layout()
    #fig.suptitle(r'Relative Area of Events for Different $\beta$', fontsize=14)

    plt.savefig('Relative area for different sigma.pdf', bbox_inches='tight')  # , dpi=300)
    plt.show()

area(alpha=0.9)

#score_area(scores,areas,lengths,alpha=0.9 , percent= 0.0002)

