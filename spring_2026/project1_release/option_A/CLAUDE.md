# CS131 Project 1 - Option A: Linear Filters & Convolution

## Overview

This is a Stanford CS131 (Computer Vision) project focused on implementing and understanding linear filters, convolution, cross-correlation, and separable filters.

## Files

| File | Purpose |
|------|---------|
| `option_a.ipynb` | Main project notebook — coding + written questions (primary deliverable) |
| `option_a_exploration.ipynb` | Exploration component — separate, later due date |
| `filters.py` | All filter implementations (student code lives here) |
| `dog.jpg`, `shelf.jpg`, `shelf_dark.jpg`, `shelf_soldout.jpg`, `template.jpg` | Input images |
| `convolved_dog.png`, `padded_dog.jpg` | Reference output images for validation |

## Project Structure

### Part 1: Convolutions
- **1.1 Commutative Property** — Prove $f*h = h*f$ (written, complete)
- **1.2 Shift Invariance** — Prove convolution is shift-invariant (written, complete)
- **1.3 Linearity** — Prove superposition property holds (written, complete)
- **1.4 Implementation** — Implement `conv_nested` and `conv_fast` in `filters.py`

### Part 2: Cross-correlation
- **2.1 Template Matching** — Implement `cross_correlation`; explain limitations of raw template matching
- **2.2 Zero-mean Cross-correlation** — Implement `zero_mean_cross_correlation` to handle brightness bias
- **2.3 Normalized Cross-correlation** — Implement `normalized_cross_correlation` to handle lighting changes

### Part 3: Separable Filters
- **3.1 Theory** — Prove $I*F = (I*F_1)*F_2$ for separable $F = F_1 F_2$ (written, complete)
- **3.2 Complexity Comparison** — Argue $O(M_1 N_1 M_2 N_2)$ vs $O(M_1 N_1 (M_2+N_2))$; decompose 5x5 Gaussian into `k1` (5,1) and `k2` (1,5)

## Implementations in `filters.py`

| Function | Description |
|----------|-------------|
| `conv_nested(image, kernel)` | 4-loop naive convolution; zero-boundary handling |
| `zero_pad(image, pad_height, pad_width)` | Adds zero padding using `np.pad` |
| `conv_fast(image, kernel)` | Efficient convolution: zero-pad + flip kernel + 2-loop element-wise multiply |
| `cross_correlation(f, g)` | Calls `conv_fast(f, np.flip(g))` |
| `zero_mean_cross_correlation(f, g)` | Subtracts `np.mean(g)` before cross-correlation |
| `normalized_cross_correlation(f, g)` | Normalizes both template and each image patch (mean/std) before dot product |

## Key Concepts

- **Convolution** flips the kernel before sliding; **cross-correlation** does not.
- Zero-mean cross-correlation removes brightness bias in template matching.
- Normalized cross-correlation is robust to lighting changes by normalizing pixel intensities.
- Separable filters allow sequential 1D convolutions, reducing complexity from $O(M_2 N_2)$ per pixel to $O(M_2 + N_2)$.

## Dependencies

```python
numpy
matplotlib
skimage (scikit-image)
```

## Running the Notebook

Open `option_a.ipynb` in Jupyter and run cells top to bottom. The notebook imports functions from `filters.py` using `%autoreload`, so edits to `filters.py` are picked up automatically.
