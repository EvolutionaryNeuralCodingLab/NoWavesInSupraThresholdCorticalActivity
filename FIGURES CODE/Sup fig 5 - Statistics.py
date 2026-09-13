import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import itertools
from matplotlib.ticker import PercentFormatter


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

def process_data(data, min_area, offset,fps=60):
    scores, areas, durations, energies, intervals = [], [], [], [], []

    for i, video in enumerate(data):
        last_end = -1  # To track the end of the last valid interval

        for interval, values in sorted(data[video].items()):
            start, end = interval[0] + offset * i, interval[1] + offset * i
            if values['area'] > min_area and start >= last_end and values['duration']/fps<20:
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


min_area=-0.1
offset = 3700

dataa = True
if dataa == True:
    data_54MRL = np.load('Fig 2 - statistics/54MRL_statistics.npy',allow_pickle=True).item()
    data_63MR = np.load('Fig 2 - statistics/63MR_statistics.npy',allow_pickle=True).item()
    data_187FN = np.load('Fig 2 - statistics/187FN_statistics.npy',allow_pickle=True).item()
    data_203MN = np.load('Fig 2 - statistics/203MN_statistics.npy',allow_pickle=True).item()
    data_204FR = np.load('Fig 2 - statistics/204FR_statistics.npy',allow_pickle=True).item()
    data_206FRL = np.load('Fig 2 - statistics/206FRL_statistics.npy',allow_pickle=True).item()
    data_211MRR = np.load('Fig 2 - statistics/211MRR_statistics.npy',allow_pickle=True).item()
    data_218MN = np.load('Fig 2 - statistics/218MN_statistics.npy',allow_pickle=True).item()
    data_21ML = np.load('Fig 2 - statistics/21ML_statistics.npy',allow_pickle=True).item()
    data_221ML = np.load('Fig 2 - statistics/221ML_statistics.npy',allow_pickle=True).item()

    score_54MRL, area_54MRL, duration_54MRL, energy_54MRL, intervals_54MRL = process_data(data_54MRL, min_area, offset)
    score_63MR, area_63MR, duration_63MR, energy_63MR, intervals_63MR = process_data(data_63MR, min_area, offset)
    score_187FN, area_187FN, duration_187FN, energy_187FN, intervals_187FN = process_data(data_187FN, min_area, offset)
    score_203MN, area_203MN, duration_203MN, energy_203MN, intervals_203MN = process_data(data_203MN, min_area, offset)
    score_204FR, area_204FR, duration_204FR, energy_204FR, intervals_204FR = process_data(data_204FR, min_area, offset)
    score_206FRL, area_206FRL, duration_206FRL, energy_206FRL, intervals_206FRL = process_data(data_206FRL, min_area,                                                                                              offset)
    score_211MRR, area_211MRR, duration_211MRR, energy_211MRR, intervals_211MRR = process_data(data_211MRR, min_area,                                                                                               offset)
    score_218MN, area_218MN, duration_218MN, energy_218MN, intervals_218MN = process_data(data_218MN, min_area, offset)
    score_21ML, area_21ML, duration_21ML, energy_21ML, intervals_21ML = process_data(data_21ML, min_area, offset)

    score_221ML, area_221ML, duration_221ML, energy_221ML, intervals_221ML = process_data(data_221ML, min_area, offset)

    scores = [score_54MRL, score_63MR, score_187FN, score_203MN, score_204FR, score_206FRL, score_211MRR, score_218MN,score_21ML, score_221ML]
    areas = [area_54MRL, area_63MR, area_187FN, area_203MN, area_204FR, area_206FRL, area_211MRR, area_218MN, area_21ML,area_221ML]
    durations = [duration_54MRL, duration_63MR, duration_187FN, duration_203MN, duration_204FR, duration_206FRL,duration_211MRR, duration_218MN, duration_21ML, duration_221ML]
    energies = [energy_54MRL, energy_63MR, energy_187FN, energy_203MN, energy_204FR, energy_206FRL,energy_211MRR, energy_218MN, energy_21ML, energy_221ML]



scores = list(itertools.chain(*scores))


figsize_cm = (30, 9)  # example in cm
figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches


linewidth = 2
font_size = 8
face_color = "#1f1f2e"
face_color_2 = "#dee2e6"
face_color_3 = "#f8f9fa"
face_text_color = "#f8f9fa"

line_color = "#1f1f2e"
dashed_line_color = "#d80000"
bins_color = "#adb5bd"


colWidths= [0.32, 0.15, 0.36]
bb_box = [-0.2, 0, 1.3, 1.3]


fig, axs = plt.subplots(2, 5, figsize=figsize_in)   # <-- adjust number of subplots here
ax1 = axs[0,0]

sns.set_style("white")
sns.set_context("paper", font_scale=1.35)


a1, b1 = 0.062126 * 16.870, (1 - 0.062126) * 16.870
a3, b3 = 0.06240 * 20.028, (1 - 0.06240) * 20.028

bins_scores = np.linspace(0, 1, 11)
weights = np.ones_like(scores) * 100.0 / len(scores)
x = np.linspace(0, 1, 100)
bin_width = bins_scores[1] - bins_scores[0]


ax1.hist(scores, bins=bins_scores, weights=weights, color=bins_color, edgecolor="white", linewidth=1.5, alpha=1.0,label=f'Observed scores')

# beta models scaled to percent
y_beta_scaled = beta.pdf(x, a1, b1) * bin_width * 100
ax1.plot(x, y_beta_scaled, color=line_color, lw=linewidth, label="Baseline model")

y_beta_scaled = beta.pdf(x, a3, b3) * bin_width * 100
ax1.plot(x, y_beta_scaled, color=dashed_line_color, lw=linewidth, ls="--",label="Predictiors model")

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 100)
ax1.set_xlabel(f"Waviness score, N = {len(scores)}")
ax1.set_ylabel("Events [%]")
ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
sns.despine(ax=ax1, trim=True)


ax4 = axs[1,0]

table_data = [
    ["Parameter", "Mean", "95% interval"],
    ["Score", "0.058", "[0.056,0.060]"],
    ["Precision", "11.42", "[10.78,12.08]"],
    ["Score",      "0.057", "[0.030,0.079]"],
    ["Precision",   "14.32", "[13.54, 15.15]"],
    [r"$\beta_0$",   "-2.83", "[-2.88,-2.80]"],
    ["Area",       "-0.11", "[-0.18,-0.04]"],
    ["Length",     "-0.073", "[-0.10,-0.04]"],
    ["Avg Intensity",     "-0.14", "[-0.21,-0.07]"],
]


ax4.axis('off')

table = ax4.table(
    cellText=table_data,
    colWidths=colWidths,
    cellLoc='center',
    bbox=bb_box
)

table.auto_set_font_size(False)
table.set_fontsize(font_size)  # slightly larger since it's full size

# Styling rows
for (r, c), cell in table.get_celld().items():
    # Header row
    if r == 0:
        cell.set_facecolor(face_color)
        cell.set_text_props(color=face_text_color, weight="bold")
    elif r ==1 or r==2:
        if r==1:
            cell.set_text_props(weight="bold")
        cell.set_facecolor(face_color_2)
    else:
        if r==3:
            cell.set_text_props(weight="bold")
        cell.set_facecolor(face_color_3)


x0, y0, w, h = bb_box

ax4.text(
    x0 - 0.25,      # slightly left of the table
    y0 + h - 0.275,  # around the height of rows 1–2
    "Baseline\nModel",
    va='center',
    ha='center',
    fontsize=font_size,
    fontweight='bold'
)

# Position of Experiment B (rows 3 to end)
ax4.text(
    x0 - 0.25,
    y0 + h - 0.575,  # adjust so it aligns with middle of lower block
    "Predictors\nModel",
    va='center',
    ha='center',
    fontsize=font_size,
    fontweight='bold'
)





data = np.load(f'Sup fig 5/murphy_processed_data.npz')

scores = data['scores']
areas = data['areas']
durations = data['durations']
energies = data['energies']

print(len(scores))


ax2 = axs[0,1]


a1, b1 = 0.062126 * 16.870, (1 - 0.062126) * 16.870   # → a1 ≈ 1.048, b1 ≈ 15.822

# NEW full 3-covariate model (red dashed line)
mu_full_mean  = 0.08656
phi_full_mean = 5.427
a3 = mu_full_mean * phi_full_mean        # ≈ 0.4697
b3 = (1 - mu_full_mean) * phi_full_mean  # ≈ 4.9573

bins_scores = np.linspace(0, 1, 11)
weights = np.ones_like(scores) * 100.0 / len(scores)
x = np.linspace(0, 1, 100)
bin_width = bins_scores[1] - bins_scores[0]





ax2.hist(scores, bins=bins_scores, weights=weights,color=bins_color, edgecolor="white", linewidth=1.2, alpha=1.0,label=f'Observed scores (N = {len(scores):,})')

# beta models scaled to percent
y_beta_scaled = beta.pdf(x, a1, b1) * bin_width * 100
ax2.plot(x, y_beta_scaled, color=line_color, lw=linewidth, label="Intercept-only model")

y_beta_scaled = beta.pdf(x, a3, b3) * bin_width * 100
ax2.plot(x, y_beta_scaled, color=dashed_line_color, lw=linewidth, ls="--",
         label="Full model (+ length, duration, energy)")

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 100)
ax2.set_xlabel(f"Waviness score, N = {len(scores)}")
#ax2.set_ylabel("Percentage of events (%)")
ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
sns.despine(ax=ax2, trim=True)



handles, labels = axs[0,0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc='lower center',  # or 'upper center', depending on your layout
    bbox_to_anchor=(0.55, 0.975),  # center horizontally, slightly below the figure
    frameon=False,
    fontsize=8,
    ncol=3,  # <-- this makes it a single row
    handlelength=2.3
)



ax5 = axs[1,1]


table_data = [
    ["Parameter", "Mean", "95% interval"],
    ["Score", "0.075", "[0.058,0.096]"],
    ["Precision", "3.39", "[2.82,4.01]"],
    ["Score",      "0.054", "[0.023,0.099]"],
    ["Precision",   "6.52", "[5.57,7.56]"],
    [r"$\beta_0$",      "-2.95", "[-3.08,-2.81]"],
    ["Area",       "-0.22", "[-0.33,-0.12]"],
    ["Length",     "-0.16", "[-0.24,-0.08]"],
    ["Avg Intensity",     "0.48", "[0.38,0.58]"],
]

ax5.axis('off')

table = ax5.table(
    cellText=table_data,
    colWidths=colWidths,
    cellLoc='center',
    bbox=bb_box
)

table.auto_set_font_size(False)
table.set_fontsize(font_size)  # slightly larger since it's full size

# Styling rows
for (r, c), cell in table.get_celld().items():
    # Header row
    if r == 0:
        cell.set_facecolor(face_color)
        cell.set_text_props(color=face_text_color, weight="bold")
    elif r ==1 or r==2:
        if r==1:
            cell.set_text_props(weight="bold")
        cell.set_facecolor(face_color_2)
    else:
        if r==3:
            cell.set_text_props(weight="bold")
        cell.set_facecolor(face_color_3)


ax3 = axs[0,2]


### WHISKER DATA

scores = np.load('Fig 4 - datasets/whisker_waviness_scores.npy')


print(len(scores))

# 1. Intercept-only model (keep for comparison — black solid line)
a1, b1 = 0.062126 * 16.870, (1 - 0.062126) * 16.870
# → a1 ≈ 1.048, b1 ≈ 15.822 → very peaked at ~0.062

# 2. NEW FULL MODEL — your latest fit (red dashed line)
mu_full_mean  = 0.07039      # posterior mean of marginal μ
phi_full_mean = 10.043       # posterior mean of φ

a3 = mu_full_mean * phi_full_mean          # ≈ 0.707
b3 = (1 - mu_full_mean) * phi_full_mean     # ≈ 9.336

# 95% credible interval of marginal μ (for table + optional shading)
mu_full_lower = 0.0283
mu_full_upper = 0.1346

bins_scores = np.linspace(0, 1, 11)
weights = np.ones_like(scores) * 100.0 / len(scores)
x = np.linspace(0, 1, 100)
bin_width = bins_scores[1] - bins_scores[0]



ax3.hist(scores, bins=bins_scores, weights=weights, color=bins_color, edgecolor="white", linewidth=1.2, alpha=1.0, label=f'Observed scores (N = {len(scores):,})')

# beta models scaled to percent
y_beta_scaled = beta.pdf(x, a1, b1) * bin_width * 100
ax3.plot(x, y_beta_scaled, color=line_color, lw=linewidth, label="Intercept-only model")

y_beta_scaled = beta.pdf(x, a3, b3) * bin_width * 100
ax3.plot(x, y_beta_scaled, color=dashed_line_color, lw=linewidth, ls="--",
         label="Full model (+ length, duration, energy)")

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 100)
ax3.set_xlabel(f"Waviness score, N = {len(scores)}")
#ax2.set_ylabel("Percentage of events (%)")
ax3.yaxis.set_major_formatter(PercentFormatter(decimals=0))
sns.despine(ax=ax3, trim=True)


ax6 = axs[1,2]
ax6.axis('off')


table_data = [
    ["Parameter", "Mean", "95% interval"],
    ["Score", "0.067", "[0.05,0.091]"],
    ["Precision", "6.68", "[4.38,9.44]"],
    ["Score",      "0.068", "[0.027,0.135]"],
    ["Precision",       "9.50", "[6.52,13.03]"] ,
    [f"$β_0$",     "-2.69",  "[-2.95,-2.41]"],
    ["Area",        "0.42", "[0.13,0.69]"],
    ["Length",      "-0.31", "[-0.54,-0.09]"],
    ["Avg Intensity",       "-0.30", "[-0.54,-0.06]"],
]

table = ax6.table(
    cellText=table_data,
    colWidths=colWidths,
    cellLoc='center',
    bbox=bb_box
)

table.auto_set_font_size(False)
table.set_fontsize(font_size)

for (r, c), cell in table.get_celld().items():
    # Header row
    if r == 0:
        cell.set_facecolor(face_color)
        cell.set_text_props(color=face_text_color, weight="bold")
    elif r ==1 or r==2:
        if r==1:
            cell.set_text_props(weight="bold")
        cell.set_facecolor(face_color_2)
    else:
        if r==3:
            cell.set_text_props(weight="bold")
        cell.set_facecolor(face_color_3)






# ============================================================
# VSD FILES
# ============================================================
vsd_data = np.load(
    "Sup fig 5/vsd_compact.npz"
)

fully_awake_scores = vsd_data["fully_awake_scores"]
fully_awake_areas = vsd_data["fully_awake_areas"]
fully_awake_durations = vsd_data["fully_awake_durations"]

anesthetized_scores = vsd_data["anesthetized_scores"]
anesthetized_areas = vsd_data["anesthetized_areas"]
anesthetized_durations = vsd_data["anesthetized_durations"]

print("Fully awake events:", len(fully_awake_scores))
print("Anesthetized events:", len(anesthetized_scores))

print("Fully awake events:", len(fully_awake_scores))
print("Anesthetized events:", len(anesthetized_scores))



def plot_score_distribution(
    ax,
    scores,
    mu_baseline,
    phi_baseline,
    mu_predictors=None,
    phi_predictors=None,
    title=None,
):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    bins_scores = np.linspace(0, 1, 11)
    weights = np.ones(len(scores)) * 100.0 / len(scores)

    # Avoid x=0 because beta density can diverge when alpha < 1.
    x = np.linspace(0.0001, 0.9999, 500)
    bin_width = bins_scores[1] - bins_scores[0]

    ax.hist(
        scores,
        bins=bins_scores,
        weights=weights,
        color=bins_color,
        edgecolor="white",
        linewidth=1.2,
        alpha=1.0,
        label="Observed scores",
    )

    alpha_baseline = mu_baseline * phi_baseline
    beta_baseline = (1 - mu_baseline) * phi_baseline

    y_baseline = (
        beta.pdf(x, alpha_baseline, beta_baseline)
        * bin_width
        * 100
    )

    ax.plot(
        x,
        y_baseline,
        color=line_color,
        lw=linewidth,
        label="Baseline model",
    )

    if mu_predictors is not None and phi_predictors is not None:
        alpha_predictors = mu_predictors * phi_predictors
        beta_predictors = (1 - mu_predictors) * phi_predictors

        y_predictors = (
            beta.pdf(x, alpha_predictors, beta_predictors)
            * bin_width
            * 100
        )

        ax.plot(
            x,
            y_predictors,
            color=dashed_line_color,
            lw=linewidth,
            ls="--",
            label="Predictors model",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"Waviness score, N = {len(scores):,}")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    if title is not None:
        ax.set_title(title, fontsize=9)

    sns.despine(ax=ax, trim=True)

def add_statistics_table(
    ax,
    baseline_mu,
    baseline_mu_interval,
    baseline_phi,
    baseline_phi_interval,
    predictors_mu=None,
    predictors_mu_interval=None,
    predictors_phi=None,
    predictors_phi_interval=None,
    coefficients=None,
):
    ax.axis("off")

    table_data = [
        ["Parameter", "Mean", "95% interval"],
        [
            "Score",
            f"{baseline_mu:.3f}",
            f"[{baseline_mu_interval[0]:.3f}, "
            f"{baseline_mu_interval[1]:.3f}]",
        ],
        [
            "Precision",
            f"{baseline_phi:.2f}",
            f"[{baseline_phi_interval[0]:.2f}, "
            f"{baseline_phi_interval[1]:.2f}]",
        ],
    ]

    if predictors_mu is not None:
        table_data.extend([
            [
                "Score",
                f"{predictors_mu:.3f}",
                f"[{predictors_mu_interval[0]:.3f}, "
                f"{predictors_mu_interval[1]:.3f}]",
            ],
            [
                "Precision",
                f"{predictors_phi:.2f}",
                f"[{predictors_phi_interval[0]:.2f}, "
                f"{predictors_phi_interval[1]:.2f}]",
            ],
        ])

    if coefficients is not None:
        for parameter, mean, lower, upper in coefficients:
            table_data.append([
                parameter,
                f"{mean:.2f}",
                f"[{lower:.2f}, {upper:.2f}]",
            ])

    table = ax.table(
        cellText=table_data,
        colWidths=colWidths,
        cellLoc="center",
        bbox=bb_box,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(face_color)
            cell.set_text_props(
                color=face_text_color,
                weight="bold",
            )

        elif row in [1, 2]:
            cell.set_facecolor(face_color_2)

            if row == 1:
                cell.set_text_props(weight="bold")

        else:
            cell.set_facecolor(face_color_3)

            if row == 3:
                cell.set_text_props(weight="bold")

# ============================================================
# FULLY AWAKE VSD
# ============================================================

ax7 = axs[0, 3]
ax8 = axs[1, 3]

# Intercept-only model
awake_mu_baseline = 0.136
awake_mu_baseline_interval = (0.131, 0.141)

awake_phi_baseline = 5.09
awake_phi_baseline_interval = (4.81, 5.37)

# Predictors model
awake_mu_predictors = 0.138
awake_mu_predictors_interval = (0.134, 0.143)

awake_phi_predictors = 7.16
awake_phi_predictors_interval = (6.77, 7.55)

plot_score_distribution(
    ax=ax7,
    scores=fully_awake_scores,

    mu_baseline=awake_mu_baseline,
    phi_baseline=awake_phi_baseline,

    mu_predictors=awake_mu_predictors,
    phi_predictors=awake_phi_predictors,

    title="Fully awake VSD",
)

add_statistics_table(
    ax=ax8,

    baseline_mu=awake_mu_baseline,
    baseline_mu_interval=awake_mu_baseline_interval,

    baseline_phi=awake_phi_baseline,
    baseline_phi_interval=awake_phi_baseline_interval,

    predictors_mu=awake_mu_predictors,
    predictors_mu_interval=awake_mu_predictors_interval,

    predictors_phi=awake_phi_predictors,
    predictors_phi_interval=awake_phi_predictors_interval,

    coefficients=[
        ("β₀", -1.94, -1.97, -1.90),
        ("Area", -0.46, -0.51, -0.41),
        ("Length", 0.00, -0.03, 0.03),
        ("Energy", 0.78, 0.72, 0.83),
    ],
)

# ============================================================
# ANESTHETIZED VSD
# ============================================================

ax9 = axs[0, 4]
ax10 = axs[1, 4]

# Intercept-only model
anesthesia_mu_baseline = 0.319
anesthesia_mu_baseline_interval = (0.313, 0.325)

anesthesia_phi_baseline = 10.57
anesthesia_phi_baseline_interval = (9.92, 11.23)

# Predictors model
anesthesia_mu_predictors = 0.319
anesthesia_mu_predictors_interval = (0.314, 0.325)

anesthesia_phi_predictors = 11.84
anesthesia_phi_predictors_interval = (11.11, 12.59)

plot_score_distribution(
    ax=ax9,
    scores=anesthetized_scores,

    mu_baseline=anesthesia_mu_baseline,
    phi_baseline=anesthesia_phi_baseline,

    mu_predictors=anesthesia_mu_predictors,
    phi_predictors=anesthesia_phi_predictors,

    title="Anesthetized VSD",
)

add_statistics_table(
    ax=ax10,

    baseline_mu=anesthesia_mu_baseline,
    baseline_mu_interval=anesthesia_mu_baseline_interval,

    baseline_phi=anesthesia_phi_baseline,
    baseline_phi_interval=anesthesia_phi_baseline_interval,

    predictors_mu=anesthesia_mu_predictors,
    predictors_mu_interval=anesthesia_mu_predictors_interval,

    predictors_phi=anesthesia_phi_predictors,
    predictors_phi_interval=anesthesia_phi_predictors_interval,

    coefficients=[
        ("β₀", -0.76, -0.79, -0.74),
        ("Area", 0.05, 0.02, 0.08),
        ("Length", -0.08, -0.11, -0.05),
        ("Energy", -0.23, -0.26, -0.20),
    ],
)

fig.tight_layout()
plt.savefig("Sup_fig_Statistics.pdf", dpi=500, bbox_inches="tight")   # PNG

plt.show()