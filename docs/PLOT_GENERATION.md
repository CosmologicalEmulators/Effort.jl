# Documentation Plot Generation Guide

This guide explains how to manage the static plots used in Effort.jl documentation.

## Quick Start

Generate all documentation plots:

```bash
julia --project=docs docs/generate_plots.jl
```

Output: `docs/src/assets/plots/*.png` (3 files, ~520 KB total)

## Why Static Plots?

Using pre-generated static plots instead of dynamically generating them during documentation builds provides several benefits:

1. **Faster CI/CD**: No need to run expensive emulator computations during every doc build
2. **Consistent appearance**: Plots look identical across all documentation builds
3. **Version control**: Track visual changes to documentation plots
4. **Reliability**: Avoids plot generation failures during CI

## Generated Plots

The script generates three publication-quality plots (300 DPI, LaTeX fonts):

### 1. `multipoles_no_ap.png` (~180 KB)
- Power spectrum multipoles (ℓ = 0, 2, 4) at z = 0.8
- Shows the characteristic shape of galaxy clustering
- Used in Step 5 of the tutorial

### 2. `ap_effect_comparison.png` (~200 KB)
- Before/after comparison of Alcock-Paczynski corrections
- Demonstrates how cosmological parameter differences affect observables
- Used in Step 6 of the tutorial

### 3. `ap_relative_difference.png` (~140 KB)
- Relative percent difference showing AP effect magnitude
- Highlights where AP corrections are most important (low k)
- Additional figure showing quantitative impact

## Cosmology Setup

To demonstrate visible AP effects, we use **intentionally different** cosmologies:

| Parameter | Fiducial | Reference | Δ (%) |
|-----------|----------|-----------|-------|
| h         | 0.6736   | 0.7000    | +3.9% |
| ωc        | 0.1200   | 0.1150    | -4.2% |
| w0        | -1.0000  | -0.9500   | +5.0% |

**Result**: q_∥ ≈ 0.98, q_⊥ ≈ 0.99 (clearly visible AP effect)

## Plot Styling

All plots use consistent, publication-ready styling:

```julia
# LaTeX-style fonts and formatting
fontfamily = "Computer Modern"
guidefont = (14, :black)    # Axis labels
tickfont = (12, :black)     # Tick labels
legendfont = (11, :black)   # Legend text

# Professional appearance
framestyle = :box
grid = false
minorticks = true
linewidth = 2.5
```

## Workflow

### For Package Maintainers

When updating plots (e.g., after changing emulator training, cosmology defaults, etc.):

1. **Update parameters** in `docs/generate_plots.jl` if needed
2. **Regenerate plots**: `julia --project=docs docs/generate_plots.jl`
3. **Review plots** visually to ensure they look correct
4. **Commit updated PNGs** to version control
5. **Documentation automatically uses** the updated plots

### For Cloud Deployment (Optional)

If you want to host plots externally to reduce repository size:

1. Upload PNGs to cloud storage (GitHub releases, Zenodo, CDN)
2. Update image URLs in `docs/src/example.md`:
   ```markdown
   ![Description](https://your-cdn.com/path/to/image.png)
   ```
3. Add `.png` to `.gitignore` for `docs/src/assets/plots/`

**Current setup**: Plots are committed to the repository (recommended for simplicity).

## Troubleshooting

### "Package X not found"

Install documentation dependencies:
```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

### Plots look different from expected

Ensure you're using the correct versions:
```bash
julia --project=docs -e 'using Pkg; Pkg.status(["Effort", "Plots", "LaTeXStrings"])'
```

### High memory usage during generation

The script loads emulators and runs computations. Typical memory usage: ~2 GB.

### Want higher/lower resolution?

Edit `dpi` parameter in `generate_plots.jl`:
- 150 DPI: smaller files, web-friendly
- 300 DPI: current setting, publication-quality
- 600 DPI: very high quality, large files

## Integration with Documentation Build

The plots are automatically included during documentation builds:

1. `docs/make.jl` runs `makedocs(...)`
2. Documenter.jl copies `docs/src/assets/` to `docs/build/assets/`
3. Markdown image links `![](assets/plots/X.png)` resolve correctly
4. No dynamic plot generation occurs

This approach separates **content generation** (plots) from **documentation rendering** (HTML).

## File Locations

```
Effort.jl/
├── docs/
│   ├── generate_plots.jl              # Plot generation script
│   ├── make.jl                         # Documentation builder
│   └── src/
│       ├── example.md                  # References static plots
│       └── assets/
│           └── plots/
│               ├── README.md           # This directory's README
│               ├── multipoles_no_ap.png
│               ├── ap_effect_comparison.png
│               └── ap_relative_difference.png
```

## Version History

- **v0.4.1** (2025-01-26): Initial static plot generation setup
  - Separated plot generation from documentation build
  - Enhanced AP effect visibility (q_∥ = 0.98, q_⊥ = 0.99)
  - Added relative difference plot
  - Static benchmark generation using BenchmarkTools native format

---

**Questions?** See `docs/src/assets/plots/README.md` or open an issue.

## Benchmark Generation

Similar to plot generation, benchmarks are computed locally and saved for documentation builds.

**Generate benchmarks:**
```bash
julia --project=docs docs/run_benchmarks.jl
```

Output: `docs/src/assets/effort_benchmark.json` (BenchmarkTools native format)

This avoids expensive emulator computations during CI/CD builds.
