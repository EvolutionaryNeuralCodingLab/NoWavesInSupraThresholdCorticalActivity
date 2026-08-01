# Wave Propagation Analysis

This repository analyzes spatiotemporal imaging data to determine whether an activity event exhibits propagating-wave dynamics. The analysis combines:

1. Horn–Schunck optical-flow estimation.
2. Intensity-weighted flow, referred to as the momentum field.
3. Wavefront-profile detection.
4. Temporal directional coherence.
5. A final waviness map and event-level waviness score.

The main script can be used with synthetic patterns or with experimental imaging data stored as a three-dimensional NumPy array.

---

## 1. Expected data format

The analysis expects a NumPy array with the following shape:

```python
(N, M, T)
```

where:

- `N` is the image height in pixels.
- `M` is the image width in pixels.
- `T` is the number of time frames.

The script currently assumes that the activity values are normalized approximately to the range `[0, 1]`.

Example:

```python
dff1 = np.load("path/to/event.npy")
print(dff1.shape)  # Example: (128, 64, 50)
```

The input should represent one activity event rather than an entire unsegmented recording.

---

## 2. Repository structure

The script imports several project-specific modules:

```text
project_root/
├── analysis.py
├── Algos/
│   ├── __init__.py
│   ├── Data_Processing.py
│   ├── Create_Patterns.py
│   ├── Display.py
│   └── Horn_Schunck.py
├── brain_mask.npy
└── outer_line_rgb.npy
```

The following functions must be available:

```python
from Algos.Data_Processing import Filter, resize
from Algos.Create_Patterns import (
    create_patterns,
    create_gaussians,
    create_gaussians_moving,
)
from Algos.Display import plot_quiver
from Algos.Horn_Schunck import horn_schunck, horn_schunck_phase
```


---

## 3. Installation

A suitable environment can be created with Python 3.10 or later.

Install the required public packages:

```bash
pip install numpy scipy matplotlib opencv-python
```

Alternatively, create a `requirements.txt` file containing:

```text
numpy
scipy
matplotlib
opencv-python
```

Then run:

```bash
pip install -r requirements.txt
```

The custom `Algos` package must also be present in the project directory or installed in the active Python environment.

---

## 4. Required mask and outline files

The script loads two local files:

### `brain_mask.npy`

A two-dimensional 128x128 binary mask indicating pixels that belong to the cortex.

Expected values:

```text
0 = outside the analyzed tissue
1 = inside the analyzed tissue
```

Its spatial dimensions must match the analyzed data.

### `outer_line_rgb.npy`

An 128x128 RGB image used as an anatomical or cortical outline in the generated figures.

Its first two dimensions should match the spatial dimensions of the displayed data.

The original script uses hard-coded paths such as:

```python
brain_mask = np.load(
    "/Users/arielrom/Desktop/.../brain_mask.npy"
)
```

Replace every hard-coded path with a path valid on your machine. A recommended approach is to define the paths once near the top of the script:

```python
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BRAIN_MASK_PATH = PROJECT_DIR / "brain_mask.npy"
OUTER_LINE_PATH = PROJECT_DIR / "outer_line_rgb.npy"
```

Then replace the original loading commands with:

```python
brain_mask = np.load(BRAIN_MASK_PATH)
outer_line_rgb = np.load(OUTER_LINE_PATH)
```

---

## 5. Main analysis parameters

The principal parameters are defined near the top of the script:

```python
alpha = 0.3
iterations = 150
n = 3
lim = 0.5
gamma = 0.035
```

### `alpha`

Horn–Schunck smoothness regularization parameter.

- Lower values permit more spatial variation in the optical-flow field.
- Higher values produce smoother flow fields.

Current value:

```python
alpha = 0.3
```

### `iterations`

Number of Horn–Schunck optimization iterations for every pair of consecutive frames.

Current value:

```python
iterations = 150
```

Increasing this value may improve convergence but will increase execution time.

### `n`

Radius of the local neighborhood used when spatially averaging the cumulative momentum field.

Current value:

```python
n = 3
```

This corresponds approximately to a local neighborhood of size:

```text
(2n + 1) × (2n + 1)
```

### `lim`

Threshold applied to the pixel-level waviness values.

Pixels with waviness greater than `lim` contribute to the final event-level score.

Current value:

```python
lim = 0.5
```

### `gamma`

Relative variance-threshold coefficient used to determine active pixels.

The script currently applies:

```python
variance_threshold = gamma * 0.25
```

For data normalized to `[0, 1]`, `0.25` is the maximum possible temporal variance. Therefore:

```python
gamma = 0.035
```

corresponds to:

```python
0.035 * 0.25
```

The threshold should be reconsidered if the data range or normalization method changes.

---

## 6. Analysis workflow

The standard workflow is:

```python
data = PreDataProcessing(dff1)

analysis = FlowAnalyze(data)
analysis.horn_schunck_flow(
    alpha=alpha,
    num_iter=iterations,
    phase=False,
)
analysis.calculate_waveness(type="retina")

display = Display(analysis)
display.title = "example"
display.plot_data()
display.full_analysis(
    space=5,
    scale=0.15,
    data_type="retina",
)
```

---

## 7. Running synthetic examples

The function `data_type()` generates several synthetic activity patterns.

Example:

```python
dff1, title = data_type("plane")
```

Available options in the supplied script include:

```text
1 gaussian
1 gaussian moving
2 gaussian 2.25sig
3 gaussian
2 gaussian moving
plane
radial
radial with gaps
spiral
2 diff gaussians moving
cont
```

Run a plane-wave example:

```python
dff1, title = data_type("plane")

data = PreDataProcessing(dff1)

analysis = FlowAnalyze(data)
analysis.horn_schunck_flow(
    alpha=alpha,
    num_iter=iterations,
    phase=False,
)
analysis.calculate_waveness(type="retina")

display = Display(analysis)
display.title = title
display.full_analysis(
    space=5,
    scale=0.15,
    data_type="retina",
)
```

Run a non-propagating Gaussian example:

```python
dff1, title = data_type("1 gaussian")
```

Synthetic propagating patterns such as plane and radial waves should generally yield higher waviness scores than stationary or sequential modular activations.

---

## 8. Running experimental data

Replace the synthetic-data line with a NumPy loading command:

```python
dff1 = np.load("path/to/event.npy")
```

Confirm the input dimensions:

```python
if dff1.ndim != 3:
    raise ValueError(
        f"Expected an N × M × T array, received shape {dff1.shape}"
    )
```

A complete example is:

```python
import numpy as np

dff1 = np.load("path/to/event.npy")

if dff1.ndim != 3:
    raise ValueError("Input data must have shape N × M × T.")

data = PreDataProcessing(dff1)

analysis = FlowAnalyze(data)
analysis.horn_schunck_flow(
    alpha=0.3,
    num_iter=150,
    phase=False,
)
analysis.calculate_waveness(type="cortex")

display = Display(analysis)
display.title = "experimental_event"
display.full_analysis(
    space=5,
    scale=0.15,
    data_type="cortex",
)
```

Use:

```python
type="cortex"
data_type="cortex"
```

when applying the cortical brain mask and anatomical outline.

For non-cortical datasets, such as retina or synthetic patterns, use another label consistently:

```python
type="retina"
data_type="retina"
```

---

## 9. Optional preprocessing

The `PreDataProcessing` class provides several preprocessing operations.

### Spatial resizing

```python
data = PreDataProcessing(dff1)
data.resize(m=128, n=64)
```

This resizes every frame to:

```text
128 × 64 pixels
```

The current method name uses the arguments `(m, n)`, which correspond to the output height and width.

### Spatial filtering

```python
data.filter(
    fil="gaussian",
    sigma=0.5,
    kernel=3,
)
```

The accepted filter names and parameter behavior depend on the custom `Filter` function in `Algos/Data_Processing.py`.

### Add Gaussian noise

```python
data.add_noise(std=0.05)
```

This adds independent Gaussian noise to every voxel.

### Calculate ΔF/F

```python
data.delta_f_over_f(T1=10, T2=50)
```

For every pixel, the method:

1. Calculates a moving average of length `T1`.
2. Calculates a floating minimum baseline over a window of length `T2`.
3. Computes:

```text
ΔF/F = (F − F₀) / F₀
```

Care should be taken when the baseline is close to zero.

---

## 10. Optical-flow calculation

Run optical flow with:

```python
analysis.horn_schunck_flow(
    alpha=alpha,
    num_iter=iterations,
    phase=False,
)
```

Set:

```python
phase=False
```

to use conventional intensity-based Horn–Schunck optical flow.

Set:

```python
phase=True
```

to use `horn_schunck_phase`, provided that the custom phase-based implementation is available and appropriate for the data.

For each frame transition, the method stores:

```python
analysis.velocities
analysis.phase_space
analysis.space_and_time
analysis.flows
analysis.converge
```

The momentum field is calculated as:

```text
P(x, y, t) = I(x, y, t) · V(x, y, t)
```

where `I` is activity intensity and `V` is the optical-flow vector.

---

## 11. Waviness calculation

Run:

```python
analysis.calculate_waveness(type="cortex")
```

The calculation contains three main components.

### Active-pixel selection

Temporal variance is calculated for every pixel:

```python
variance = np.var(self.dff, axis=2)
```

A pixel is active when:

```python
variance > gamma * 0.25
```

For cortical data, the pixel must also be inside the brain mask.

### Temporal directional coherence

For every pixel, the script calculates:

```text
TC = ||Σₜ P(t)|| / Σₜ ||P(t)||
```

This ratio is bounded approximately between `0` and `1`.

- Values near `1` indicate a stable momentum direction over time.
- Values near `0` indicate cancellation or changing momentum directions.

### Wavefront detection

At each candidate pixel, the script:

1. Aligns a rectangle with the local flow direction.
2. Extracts a one-dimensional temporal-gradient profile.
3. Smooths the profile.
4. Detects a trough followed by a peak.
5. Assigns a binary wavefront value.

The default rectangle size is currently defined inside `process_data()`:

```python
rect_size = (34, 4)
```

This parameter may need adjustment when the spatial resolution or physical field of view changes.

### Pixel-level waviness

The final pixel value is:

```text
W(x, y) = wavefront(x, y) × temporal_coherence(x, y)
```

It is stored in:

```python
analysis.waveness[:, :, 3]
```

---

## 12. Event-level waviness score

The final score is printed by:

```python
display.full_analysis(...)
```

The event score is calculated as:

```text
number of active pixels with W > 0.5
-------------------------------------
total number of active pixels
```

The threshold `0.5` is controlled by:

```python
lim = 0.5
```

The score is printed to the console:

```text
Score : 0.73
```

A score closer to `1` indicates that a larger fraction of active pixels exhibit both a detected wavefront and temporally coherent directional activity.

A score closer to `0` indicates limited evidence for continuous propagating-wave dynamics.

---

## 13. Display functions

### Animate the activity

```python
display.plot_data()
```

This opens a Matplotlib animation of the input event.

The current time label assumes a frame rate of 25 frames per second:

```python
time_sec = i / 25
```

Change `25` to the actual acquisition frame rate.

### Plot selected frames

```python
display.plot_frames()
```

This plots the frame indices currently defined in the method:

```python
frame_indices = [1, 2, 3, 4, 5]
```

Modify this list to display other time points.

### Plot the full analysis

```python
display.full_analysis(
    space=5,
    scale=0.15,
    data_type="cortex",
)
```

This generates:

1. The cumulative momentum vector field.
2. A histogram of active-pixel waviness values.
3. A directional waviness map.

Parameters:

- `space`: spacing between displayed quiver arrows.
- `scale`: quiver-arrow scaling.
- `data_type`: use `"cortex"` to add the cortical outline.

The output PDF name follows:

```text
{N}x{M}x{T} {title}.pdf
```

For example:

```text
128x64x40 Plane Wave.pdf
```

---

## 14. Minimal runnable example

```python
import numpy as np

# Select or load an event.
dff1, title = data_type("plane")

# Optional preprocessing.
preprocessed = PreDataProcessing(dff1)

# Run optical flow.
analysis = FlowAnalyze(preprocessed)
analysis.horn_schunck_flow(
    alpha=0.3,
    num_iter=150,
    phase=False,
)

# Calculate the waviness map.
analysis.calculate_waveness(type="retina")

# Display and save the results.
display = Display(analysis)
display.title = title

display.full_analysis(
    space=5,
    scale=0.15,
    data_type="retina",
)
```

---

## 15. Recommended experimental-data example

```python
from pathlib import Path

import numpy as np

DATA_PATH = Path("data/event_001.npy")

dff1 = np.load(DATA_PATH)

if dff1.ndim != 3:
    raise ValueError(
        f"Input must have shape N × M × T; received {dff1.shape}."
    )

if not np.all(np.isfinite(dff1)):
    raise ValueError("Input contains NaN or infinite values.")

preprocessed = PreDataProcessing(dff1)

analysis = FlowAnalyze(preprocessed)
analysis.horn_schunck_flow(
    alpha=alpha,
    num_iter=iterations,
    phase=False,
)
analysis.calculate_waveness(type="cortex")

display = Display(analysis)
display.title = DATA_PATH.stem
display.full_analysis(
    space=5,
    scale=0.15,
    data_type="cortex",
)
```

---

## 16. Common problems

### `ModuleNotFoundError: No module named 'Algos'`

Run the script from the repository root, where the `Algos` directory is located.

Also confirm that `Algos/__init__.py` exists.

### Mask shape does not match the data

Check:

```python
print(dff1.shape[:2])
print(brain_mask.shape)
```

Both spatial shapes must agree.

The supplied code contains special handling for data with width `64`:

```python
if self.dff.shape[1] == 64:
    brain_mask = brain_mask[:, :64]
```

Verify that this selects the intended hemisphere.

### The analysis is very slow

Horn–Schunck flow is calculated separately for every consecutive frame pair.

Possible ways to reduce runtime include:

- Analyze shorter events.
- Reduce the spatial dimensions.
- Reduce `iterations`.
- Process multiple events in parallel outside this class.
- Avoid repeatedly loading masks inside analysis loops.

### The score is always zero

Check:

1. Whether any pixels pass the variance threshold.
2. Whether the mask matches the data.
3. Whether the data were normalized appropriately.
4. Whether the rectangle size is appropriate.
5. Whether the temporal-gradient limits are too strict.
6. Whether the optical-flow field is nonzero.

Useful checks:

```python
variance = np.var(dff1, axis=2)
print("Maximum variance:", variance.max())
print(
    "Active pixels:",
    np.count_nonzero(variance > gamma * 0.25),
)
```

### The score is unexpectedly high

Check whether:

- Noise produces false temporal gradients.
- The active-pixel threshold is too permissive.
- The wavefront profile thresholds are too permissive.
- The data contain motion artifacts.
- The normalization amplifies weak fluctuations.
- The analyzed interval contains several separate events.

### PDF text is not editable

The script sets:

```python
matplotlib.rcParams["pdf.fonttype"] = 42
```

This normally preserves TrueType text in PDF output. Confirm that the requested font, currently Arial, is installed.

---

## 17. Important implementation notes

The supplied script is research code and contains several items that should be reviewed before distribution or large-scale use.

### Hard-coded local paths

Multiple mask, outline, video, and output paths are specific to one computer. Move these paths into a configuration section.

### Missing optional definitions

The spiral-data branch references:

```python
MP4ToDff
decrease_frame_rate
normalize_data
```

These are not imported in the supplied script.

### Duplicate condition

`data_type()` contains two branches named:

```python
if type == "radial with gaps":
```

Only the first matching branch can be reached. Rename or remove one branch.

### Unreachable duplicate return

`find_max_min()` currently contains:

```python
return 1
return 1
```

The second statement is unreachable and can be removed.

### Built-in name shadowing

Several functions use the name:

```python
type
```

as an argument. This shadows Python's built-in `type()` function. A clearer alternative is:

```python
data_kind
```

### Frame-rate assumption

`plot_data()` assumes 25 frames per second. The frame rate should be supplied explicitly.

### In-place modification

`full_analysis()` applies the display threshold directly to:

```python
data.waveness[:, :, 3]
```

This permanently changes the stored waviness map. Use a copied array if the original values must be retained:

```python
display_alpha = data.waveness[:, :, 3].copy()
display_alpha[display_alpha < self.lim] = 0
```

### Repeated file loading

The masks are loaded repeatedly inside different methods. Loading them once and passing them to the relevant classes will simplify the code and improve efficiency.

### Variance scaling

The expression:

```python
gamma * 0.25
```

is valid only when the intended theoretical maximum variance is `0.25`, as for bounded `[0, 1]` data. Update it for data normalized differently, including data spanning `[-1, 1]`.

---

## 18. Suggested command-line organization

For reproducible batch use, the script can eventually be wrapped in a command-line interface such as:

```bash
python analysis.py \
    --input data/event_001.npy \
    --brain-mask brain_mask.npy \
    --outer-line outer_line_rgb.npy \
    --data-type cortex \
    --fps 25 \
    --alpha 0.3 \
    --iterations 150 \
    --variance-fraction 0.035 \
    --waviness-threshold 0.5 \
    --output results/event_001
```

The current script does not yet implement these command-line arguments.

---

## 19. Output interpretation

The analysis is intended to distinguish propagating waves from activity patterns that merely activate different spatial modules in sequence.

Interpret the output using all three panels:

- **Momentum field:** shows the accumulated direction of activity transport.
- **Waviness histogram:** shows the distribution of pixel-level waviness among active pixels.
- **Waviness map:** identifies spatial locations exhibiting a wavefront and coherent momentum direction.

The event-level score should not be interpreted in isolation. Inspect the input data, active-pixel mask, optical-flow field, and waviness map, particularly when analyzing a new imaging modality or spatial resolution.

---

## 20. Citation and research use

When using this code in a publication or shared project, document:

- Imaging modality.
- Spatial dimensions and physical pixel size.
- Acquisition frame rate.
- Event-detection procedure.
- Normalization procedure.
- Optical-flow parameters.
- Variance threshold.
- Wavefront rectangle dimensions.
- Waviness threshold.
- Any modality-specific changes.

This information is necessary to reproduce and correctly interpret the waviness score.
