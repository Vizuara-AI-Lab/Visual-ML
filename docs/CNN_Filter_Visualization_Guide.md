# CNN Filter Visualization — Complete Concept Guide

This document explains **every concept** used in the CNN Filter Visualizer activity node, tab by tab, section by section.

---

## What is This Node?

The **CNN Filter Visualizer** is an interactive teaching tool that lets you **see** how Convolutional Neural Networks (CNNs) process images. Instead of just reading theory, you can paint images, edit filters, and watch the math happen step by step.

It has **5 tabs**, each teaching a different CNN concept:

| Tab | What it Teaches |
|-----|----------------|
| Apply Filters | How a single convolution filter slides over an image |
| Edge Detection | How specialized filters find edges in images |
| Stride & Padding | How stride and padding change the output size |
| Pooling | How pooling layers shrink feature maps |
| Feature Maps & Channels | How multiple filters create multiple feature maps |

---

## Background: What is Convolution?

Before diving into tabs, here's the core idea:

**Convolution** = sliding a small grid of numbers (called a **kernel** or **filter**) across an image, and at each position, multiplying overlapping values together and summing them up to produce one output pixel.

```
Image patch:          Kernel:           Calculation:
[200  180  50]       [-1  0  1]        (200×-1) + (180×0) + (50×1)
[210  190  60]   ×   [-2  0  2]   =    (210×-2) + (190×0) + (60×2)
[205  185  55]       [-1  0  1]        (205×-1) + (185×0) + (55×1)
                                      = -200 + 0 + 50 - 420 + 0 + 120 - 205 + 0 + 55
                                      = -600
```

The kernel slides across every possible position, producing an **output grid** (called a **feature map**).

---

## Tab 1: Apply Filters

### What You See
- An **8×8 paintable image grid** (left) — each cell is a grayscale pixel (0 = black, 255 = white)
- A **3×3 kernel grid** (center) — editable numbers that define what the filter detects
- An **output grid** (right) — the result of convolution

### Concepts Used

#### 1. Pixel Grid (Input Image)
Each cell in the 8×8 grid represents one **pixel**. The number inside (0–255) is the **intensity**:
- **0** = pure black
- **255** = pure white
- Values in between = shades of gray

You can click cells to paint them, creating simple patterns to test filters on.

#### 2. Convolution Kernel (Filter)
The 3×3 grid of numbers is the **kernel**. Different number patterns detect different features:

| Kernel Name | What It Does | How |
|-------------|-------------|-----|
| **Identity** | Passes image through unchanged | Center = 1, rest = 0. Only the center pixel survives |
| **Edge (Horizontal)** | Detects horizontal edges | Top row = -1, bottom row = +1. Subtracts pixels above from pixels below — difference is large at horizontal edges |
| **Edge (Vertical)** | Detects vertical edges | Left column = -1, right column = +1. Same idea but sideways |
| **Sharpen** | Makes edges crisper | Center = 5 (amplify pixel), neighbors = -1 (subtract neighbors). Exaggerates differences |
| **Blur (Box)** | Smooths/blurs the image | All 1s. Averages the 3×3 neighborhood. Each output pixel = average of 9 neighbors |
| **Emboss** | Creates a 3D raised effect | Asymmetric weights create directional shadows |

#### 3. Element-wise Multiply + Sum
At each kernel position, the activity shows:
1. The 3×3 patch from the image
2. The 3×3 kernel
3. **Element-wise product**: each image value × corresponding kernel value
4. **Sum**: add all 9 products → this becomes one output pixel

This is the fundamental operation of convolution.

#### 4. Output Grid (Feature Map)
The result grid. For an 8×8 input with a 3×3 kernel:
- Output size = (8 - 3 + 1) × (8 - 3 + 1) = **6×6**

Colors in the output use a **diverging color scale**:
- **Blue** = negative values (the filter detected something opposite to what it looks for)
- **White** = zero (neutral — no feature detected)
- **Red/Orange** = positive values (the filter detected what it's looking for)

#### 5. Step Animation
The Play/Pause button animates the kernel sliding across the image position by position (left→right, top→bottom), showing you exactly which 3×3 patch is being processed at each step. There are 36 positions total for a 6×6 output.

---

## Tab 2: Edge Detection Deep Dive

### What You See
Four output grids side by side, each from a different edge-detection kernel applied to the same input pattern.

### Concepts Used

#### 1. Sobel-X Kernel (Vertical Edge Detector)
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```
**How it works**: Subtracts left-side pixels from right-side pixels. Where there's a sharp left-right brightness change (= a **vertical edge**), the output is large. Horizontal gradients produce zero.

**Why the 2 in the middle row?** The center row gets double weight because the pixel directly beside the current one is the most relevant for detecting an edge at that point (Gaussian-weighted importance).

#### 2. Sobel-Y Kernel (Horizontal Edge Detector)
```
[-1  -2  -1]
[ 0   0   0]
[ 1   2   1]
```
**How it works**: Same idea as Sobel-X but rotated 90°. Subtracts top from bottom. Detects **horizontal edges** (where brightness changes vertically).

#### 3. Laplacian Kernel (All-Direction Edge Detector)
```
[0   1  0]
[1  -4  1]
[0   1  0]
```
**How it works**: Compares each pixel against its 4 neighbors. If a pixel is very different from its neighbors (= it's at an edge), the output is large. Detects edges in **all directions** simultaneously.

**The math**: center pixel × (-4) + sum of 4 neighbors. If center equals the average of its neighbors, result = 0 (flat area). If center is brighter/darker than neighbors, result is non-zero (edge).

#### 4. Combined Edge Magnitude
Shows `√(Sobel_X² + Sobel_Y²)` — the **gradient magnitude**. This combines vertical and horizontal edge info into one map showing ALL edges regardless of direction.

**Why square root of sum of squares?** It's the Pythagorean theorem applied to the edge gradient. If Sobel-X finds a strong vertical edge and Sobel-Y finds a strong horizontal edge at the same pixel, the combined magnitude tells you there's a strong edge (like a corner).

#### 5. Selectable Input Patterns
Six preset patterns let you see how each kernel responds differently:
- **Square**: Strong edges on all 4 sides → Sobel-X lights up vertical sides, Sobel-Y lights up horizontal sides
- **H-Lines**: Only horizontal edges → Sobel-Y lights up, Sobel-X stays dark
- **V-Lines**: Only vertical edges → Sobel-X lights up, Sobel-Y stays dark
- **Diagonal**: Edges at 45° → both Sobel filters partially activate
- **Checker**: Edges everywhere → all kernels light up
- **Circle**: Curved edges → smooth gradient response from all kernels

---

## Tab 3: Stride & Padding

### What You See
A 10×10 input grid, a kernel, and multiple output grids showing what happens with different stride values and padding modes.

### Concepts Used

#### 1. Stride
**Stride** = how many pixels the kernel moves at each step.

- **Stride 1** (default): Kernel moves 1 pixel at a time. Maximum overlap, largest output.
- **Stride 2**: Kernel jumps 2 pixels. Output is roughly half the size.
- **Stride 3**: Kernel jumps 3 pixels. Output is roughly one-third the size.

**Why use stride > 1?** To **downsample** the feature map — reduces computation and makes the network look at larger-scale features. It's an alternative to pooling.

**Output size formula**:
```
output_size = floor((input_size - kernel_size) / stride) + 1
```
Example: 10×10 input, 3×3 kernel, stride 2:
```
output = floor((10 - 3) / 2) + 1 = floor(3.5) + 1 = 4
```

#### 2. Padding

**Valid padding** (no padding): The kernel only visits positions where it fits entirely inside the image. Output is smaller than input.

**Same padding**: Zeros are added around the image border so the output is the same size as the input (when stride=1).

```
Valid (no padding):         Same (zero padding):
Input: 10×10               Input: 10×10 → padded to 12×12
Kernel: 3×3                Kernel: 3×3
Stride: 1                  Stride: 1
Output: 8×8                Output: 10×10
```

**Why padding matters?**
- Without padding, the image **shrinks** at every layer. After many layers, it becomes tiny.
- Edge pixels are used less than center pixels (they appear in fewer kernel windows). Padding fixes this bias.
- "Same" padding preserves spatial dimensions, making network architecture design easier.

#### 3. Visual Comparison
The tab shows stride=1, 2, and 3 side by side so you can directly see how increasing stride shrinks the output, and how "same" padding keeps dimensions constant.

---

## Tab 4: Pooling Operations

### What You See
An 8×8 feature map (random values) and the results of max pooling, average pooling, and min pooling applied to it.

### Concepts Used

#### 1. What is Pooling?
Pooling is a **downsampling** operation. It divides the feature map into non-overlapping patches (e.g., 2×2) and reduces each patch to a single number. This:
- **Shrinks** the feature map (2×2 pool → half the size)
- Makes the representation **translation-invariant** (small shifts in input don't change output much)
- **Reduces computation** for subsequent layers

#### 2. Max Pooling
For each 2×2 patch, take the **maximum** value.

```
[3  1]
[4  2]  → max = 4
```

**Why max?** The maximum activation in a region means "the strongest feature detection happened somewhere in this 2×2 area." It doesn't matter exactly where — just that the feature exists.

**Most common pooling** in practice. Used in VGG, ResNet, etc.

#### 3. Average Pooling
For each 2×2 patch, take the **mean** (average).

```
[3  1]
[4  2]  → average = (3+1+4+2)/4 = 2.5
```

**Why average?** Gives a smoother, more representative summary of the region. Preserves more information about the overall intensity, but is less discriminative than max pooling.

Often used in the **final layer** of CNNs (Global Average Pooling) to collapse the entire feature map into a single value per channel.

#### 4. Min Pooling
For each 2×2 patch, take the **minimum** value.

```
[3  1]
[4  2]  → min = 1
```

**Less common** in practice. Useful when you care about the absence of features or the lowest activation in a region.

#### 5. Pool Size
The activity lets you switch between 2×2 and 3×3 pool sizes:
- **2×2**: Most common. Halves each dimension.
- **3×3**: More aggressive downsampling. Reduces each dimension by 3×.

#### 6. Pooling Stride
Usually stride = pool size (non-overlapping patches). But you can set stride < pool size for **overlapping pooling**, which preserves more spatial information at the cost of a larger output.

---

## Tab 5: Feature Maps & Channels

### What You See
A diagram showing how multiple input channels are processed by multiple kernels to create multiple output channels (feature maps).

### Concepts Used

#### 1. What Are Channels?
In a real image:
- **Grayscale** = 1 channel (just brightness)
- **RGB color** = 3 channels (Red, Green, Blue)

After the first convolution layer, each filter produces one output channel. So if you have 16 filters, the output has **16 channels** — each one is a feature map detecting a different pattern.

#### 2. Multi-Channel Convolution
When the input has C_in channels and you want C_out output channels:
- You need **C_out** filters
- Each filter has **C_in** kernels (one per input channel)
- For one output channel: convolve each kernel with its corresponding input channel, then **sum all results**

```
Total kernels = C_in × C_out
Total parameters = C_in × C_out × kernel_height × kernel_width
```

**Example**: Input has 3 channels (RGB), you want 16 output feature maps, with 3×3 kernels:
```
Total kernels = 3 × 16 = 48
Total parameters = 3 × 16 × 3 × 3 = 432
```

#### 3. Parameter Count
The activity dynamically shows the total parameter count as you adjust input/output channels. This helps you understand why deeper networks with more channels require more memory and computation.

#### 4. The SVG Diagram
Shows the flow:
```
Input Channels → [Kernel Sets] → Output Channels
     C_in      ×    C_out    =   Total Kernels
```

Each line from an input channel to an output channel represents one 3×3 kernel. This visual makes it clear why the parameter count grows as C_in × C_out.

#### 5. Why Multiple Feature Maps Matter
Each filter learns to detect a different pattern:
- Filter 1 might detect horizontal edges
- Filter 2 might detect vertical edges
- Filter 3 might detect corners
- Filter 4 might detect textures
- ... and so on

By having many filters, the network can recognize a rich set of features at each spatial location. Deeper layers combine these low-level features into higher-level concepts (edges → textures → object parts → objects).

---

## Key Mathematical Formulas Used

### Convolution Output Size
```
output_size = floor((input_size + 2×padding - kernel_size) / stride) + 1
```

### Pooling Output Size
```
output_size = floor((input_size - pool_size) / stride) + 1
```

### Edge Magnitude (Gradient)
```
magnitude = √(Sobel_X² + Sobel_Y²)
```

### Total Parameters in a Conv Layer
```
params = C_in × C_out × K_h × K_w  (+ C_out if bias is used)
```

---

## Color Coding in the Visualizer

| Color | Meaning |
|-------|---------|
| **Gray scale** (input grids) | Pixel intensity: black (0) → white (255) |
| **Blue** (output grids) | Negative convolution values |
| **White** (output grids) | Zero / neutral |
| **Red/Orange** (output grids) | Positive convolution values |
| **Yellow highlight** | Current kernel position during animation |
| **Green** | Positive kernel weights |
| **Red** | Negative kernel weights |

---

## How This Connects to Real CNNs

In a real CNN (like VGG-16 or ResNet-50), these operations stack:

```
Input Image (224×224×3)
    ↓ Conv Layer 1: 64 filters, 3×3, stride 1, same padding → 224×224×64
    ↓ ReLU activation
    ↓ Conv Layer 2: 64 filters, 3×3 → 224×224×64
    ↓ ReLU
    ↓ Max Pool 2×2, stride 2 → 112×112×64
    ↓ Conv Layer 3: 128 filters, 3×3 → 112×112×128
    ↓ ... (more layers)
    ↓ Global Average Pool → 1×1×512
    ↓ Fully Connected → class predictions
```

Each concept in this visualizer corresponds to a real layer operation:
- **Tab 1** = one convolution operation within a layer
- **Tab 2** = what early-layer filters learn automatically (edge detection)
- **Tab 3** = how stride and padding are configured per layer
- **Tab 4** = pooling layers between conv blocks
- **Tab 5** = how channels grow as you go deeper (3 → 64 → 128 → 256 → 512)

---

## Glossary

| Term | Definition |
|------|-----------|
| **Convolution** | Sliding a kernel over an image, multiplying overlapping values, and summing to produce an output |
| **Kernel / Filter** | A small grid of learnable weights (e.g., 3×3) that detects a specific pattern |
| **Feature Map** | The output grid produced by one filter — highlights where that pattern appears |
| **Stride** | How many pixels the kernel moves between positions |
| **Padding** | Adding border pixels (usually zeros) to control output size |
| **Pooling** | Downsampling by summarizing patches (max, average, or min) |
| **Channel** | One "layer" of a multi-dimensional image or feature map |
| **ReLU** | Activation function: max(0, x) — zeroes out negatives, keeps positives |
| **Sobel filter** | A specific kernel design for detecting edges with direction information |
| **Laplacian** | A kernel that detects edges in all directions by comparing center to neighbors |
| **Gradient magnitude** | Combined edge strength from both X and Y edge detectors |
| **Translation invariance** | The property that small shifts in input don't drastically change the output |
