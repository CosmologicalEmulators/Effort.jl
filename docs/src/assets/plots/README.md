# Documentation Plots

This directory contains static plots for the Effort.jl documentation.

## Generating Plots

To regenerate all plots, run:

```bash
julia --project=docs docs/generate_plots.jl
```

This will create/update the following files in this directory:
- `multipoles_no_ap.png` - Power spectrum multipoles (monopole, quadrupole, hexadecapole)
- `ap_effect_comparison.png` - Before/after Alcock-Paczynski corrections
- `ap_relative_difference.png` - Relative impact of AP effect

## Plot Specifications

All plots use:
- **Resolution**: 300 DPI (publication quality)
- **Size**: 800×500 pixels
- **Font**: Computer Modern (LaTeX style)
- **Format**: PNG with transparency

## Uploading to Cloud Storage

For CI/CD efficiency, these plots should be uploaded to cloud storage and referenced by URL in the documentation. Recommended options:

### Option 1: GitHub Release Assets

1. Create a new release or use an existing one
2. Upload the PNG files as release assets
3. Get permanent URLs like:
   ```
   https://github.com/CosmologicalEmulators/Effort.jl/releases/download/v0.4.1/multipoles_no_ap.png
   ```

### Option 2: Zenodo

1. Create a Zenodo record (same as the trained emulators)
2. Upload PNG files
3. Get DOI-backed permanent URLs

### Option 3: Repository Assets (Current Setup)

Keep plots in `docs/src/assets/plots/` and reference them relatively:
```markdown
![Description](assets/plots/filename.png)
```

**Note**: This is the current setup. Plots are version-controlled in the repository.

## Updating Documentation

If you change the plot generation script or cosmology parameters:

1. Regenerate plots: `julia --project=docs docs/generate_plots.jl`
2. Commit the updated PNG files
3. If using cloud storage, upload new versions and update URLs in `docs/src/example.md`

## Cosmology Parameters Used

### Fiducial Cosmology
- ln10^10 A_s = 3.044
- n_s = 0.9649
- h = 0.6736
- ω_b = 0.02237
- ω_c = 0.12
- Σm_ν = 0.06 eV
- w_0 = -1.0
- w_a = 0.0
- ω_k = 0.0

### Reference Cosmology (for AP effect)
- ln10^10 A_s = 3.0
- n_s = 0.96
- h = 0.70 ⭐ *different*
- ω_b = 0.022
- ω_c = 0.115 ⭐ *different*
- Σm_ν = 0.06 eV
- w_0 = -0.95 ⭐ *different (non-ΛCDM)*
- w_a = 0.0
- ω_k = 0.0

**Resulting AP parameters**: q_∥ ≈ 0.98, q_⊥ ≈ 0.99 (visible effect!)

## File Sizes

Typical sizes:
- `multipoles_no_ap.png`: ~180 KB
- `ap_effect_comparison.png`: ~200 KB
- `ap_relative_difference.png`: ~140 KB

Total: ~520 KB for all plots
