import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load Excel
# ------------------------------
df = pd.read_excel("Sup fig 8d - Waviness different alpha beta regression values.xlsx")
table_data = [df.columns.tolist()] + df.values.tolist()  # include header row

# ------------------------------
# 2. Styling parameters
# ------------------------------
font_size = 10
colWidths = [0.1, 0.1, 0.15, 0.15, 0.15]  # make first and last columns slightly wider
header_color = "#4367ae"  # strong professional blue
row_colors = ["#c5dce7", "#ffffff"]  # alternating row colors
face_text_color = "black"

# Table bounding box
bb_box = [0, 0.1, 1, 0.8]

# ------------------------------
# 3. Create figure and axis
# ------------------------------
figsize_cm = (15, 7)  # example in cm
figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches


fig, ax = plt.subplots(figsize=figsize_in)
ax.axis("off")

# ------------------------------
# 4. Create table
# ------------------------------
table = ax.table(
    cellText=table_data,
    colWidths=colWidths,
    cellLoc='center',
    bbox=bb_box
)

table.auto_set_font_size(False)
table.set_fontsize(font_size)

# ------------------------------
# 5. Style cells
# ------------------------------
for (r, c), cell in table.get_celld().items():
    # Header row
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', weight='bold', fontname='Arial')
    else:
        # Alternate row colors
        cell.set_facecolor(row_colors[r % 2])
        cell.set_text_props(color=face_text_color, fontname='Arial')
        # Bold first column
        if c == 0:
            cell.set_text_props(weight='bold', fontname='Arial')

    # Add light border lines
    cell.set_edgecolor("grey")
    cell.set_linewidth(0.5)


plt.tight_layout()
# ------------------------------
# 7. Save as PDF
# ------------------------------
plt.savefig("professional_table.pdf",
            dpi=300,
            bbox_inches='tight',
            transparent=False)


plt.show()
