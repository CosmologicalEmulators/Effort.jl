"""
Generate static plots for Effort.jl documentation.

This script generates all plots used in the documentation examples and saves them
as PNG files. These images can be uploaded to cloud storage and linked in the docs,
avoiding the need to regenerate them during every documentation build.

Usage:
    julia --project=docs docs/generate_plots.jl
"""

using Plots
using LaTeXStrings
using Effort
using Printf

# Set output directory
output_dir = joinpath(@__DIR__, "src", "assets", "plots")
mkpath(output_dir)

# Configure plot style for publication-quality figures
gr()
Plots.reset_defaults()
default(
    fontfamily = "Computer Modern",
    guidefont = (14, :black),
    tickfont = (12, :black),
    legendfont = (11, :black),
    palette = palette(:tab10),
    framestyle = :box,
    grid = false,
    minorticks = true,
    dpi = 300,  # High resolution for publication
    size = (800, 500)
)

println("=" ^ 70)
println("Generating Effort.jl Documentation Plots")
println("=" ^ 70)

#=============================================================================
Setup: Load emulators and define cosmologies
=============================================================================#

println("\n[1/5] Loading pre-trained emulators...")
monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]
k_grid = vec(monopole_emu.P11.kgrid)
println("   ✓ Emulators loaded (k-range: [$(minimum(k_grid)), $(maximum(k_grid))] h/Mpc)")

println("\n[2/5] Defining cosmologies...")
# Fiducial cosmology
cosmology = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.044,
    nₛ = 0.9649,
    h = 0.6736,
    ωb = 0.02237,
    ωc = 0.12,
    mν = 0.06,
    w0 = -1.0,
    wa = 0.0,
    ωk = 0.0
)

# Reference cosmology - MORE DIFFERENT from fiducial for visible AP effect
cosmo_ref = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.0,
    nₛ = 0.96,
    h = 0.70,         # Changed from 0.67 to 0.70 (larger difference)
    ωb = 0.022,
    ωc = 0.115,       # Changed from 0.119 to 0.115 (larger difference)
    mν = 0.06,
    w0 = -0.95,       # Changed from -1.0 to -0.95 (non-ΛCDM)
    wa = 0.0,
    ωk = 0.0
)
println("   ✓ Cosmologies defined")
println("      Fiducial: h = $(cosmology.h), ωc = $(cosmology.ωc), w0 = $(cosmology.w0)")
println("      Reference: h = $(cosmo_ref.h), ωc = $(cosmo_ref.ωc), w0 = $(cosmo_ref.w0)")

# Redshift and bias parameters
z = 0.8
bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]

# Compute growth factors
D = Effort.D_z(z, cosmology)
f = Effort.f_z(z, cosmology)
bias_params[8] = f  # Update growth rate

println("   ✓ Growth factors computed: D(z=$z) = $(round(D, digits=4)), f = $(round(f, digits=4))")

# Build emulator input
emulator_input = [
    z, cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h * 100,
    cosmology.ωb, cosmology.ωc, cosmology.mν, cosmology.w0, cosmology.wa
]

#=============================================================================
Plot 1: Power Spectrum Multipoles (without AP)
=============================================================================#

println("\n[3/5] Generating Plot 1: Power spectrum multipoles...")

# Compute multipoles
P0 = Effort.get_Pℓ(emulator_input, D, bias_params, monopole_emu)
P2 = Effort.get_Pℓ(emulator_input, D, bias_params, quadrupole_emu)
P4 = Effort.get_Pℓ(emulator_input, D, bias_params, hexadecapole_emu)

# Create plot
p1 = plot(k_grid, k_grid .* P0,
    label=L"\ell=0\ \mathrm{(Monopole)}",
    xlabel=L"k\ [h/\mathrm{Mpc}]",
    ylabel=L"k\,P_\ell(k)\ [(\mathrm{Mpc}/h)^2]",
    linewidth=2.5,
    legend=:bottomright,
    color=1
)
plot!(p1, k_grid, abs.(k_grid .* P2),
    label=L"|\ell=2|\ \mathrm{(Quadrupole)}",
    linewidth=2.5,
    color=2
)
plot!(p1, k_grid, abs.(k_grid .* P4),
    label=L"|\ell=4|\ \mathrm{(Hexadecapole)}",
    linewidth=2.5,
    color=3
)
title!(p1, L"\mathrm{Power\ Spectrum\ Multipoles\ at}\ z = 0.8")

# Save
plot1_path = joinpath(output_dir, "multipoles_no_ap.png")
savefig(p1, plot1_path)
println("   ✓ Saved: $plot1_path")

#=============================================================================
Plot 2: Alcock-Paczynski Effect Comparison
=============================================================================#

println("\n[4/5] Generating Plot 2: AP effect comparison...")

# Compute AP parameters
q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)
println("   AP parameters: q_∥ = $(round(q_par, digits=4)), q_⊥ = $(round(q_perp, digits=4))")

# Apply AP corrections
P0_AP, P2_AP, P4_AP = Effort.apply_AP(
    k_grid, k_grid, P0, P2, P4,
    q_par, q_perp, n_GL_points=8
)

# Create plot
p2 = plot(k_grid, k_grid .* P0,
    label=L"P_0\ \mathrm{(no\ AP)}",
    linewidth=2.5,
    linestyle=:dash,
    xlabel=L"k\ [h/\mathrm{Mpc}]",
    ylabel=L"k\,P_\ell(k)\ [(\mathrm{Mpc}/h)^2]",
    color=1,
    alpha=0.7
)
plot!(p2, k_grid, k_grid .* P0_AP,
    label=L"P_0\ \mathrm{(with\ AP)}",
    linewidth=2.5,
    color=1
)
plot!(p2, k_grid, abs.(k_grid .* P2),
    label=L"|P_2|\ \mathrm{(no\ AP)}",
    linewidth=2.5,
    linestyle=:dash,
    color=2,
    alpha=0.7
)
plot!(p2, k_grid, abs.(k_grid .* P2_AP),
    label=L"|P_2|\ \mathrm{(with\ AP)}",
    linewidth=2.5,
    color=2
)

title_str = @sprintf("Effect of AP Corrections (q_∥ = %.4f, q_⊥ = %.4f)", q_par, q_perp)
title!(p2, latexstring("\\mathrm{$title_str}"))

# Save
plot2_path = joinpath(output_dir, "ap_effect_comparison.png")
savefig(p2, plot2_path)
println("   ✓ Saved: $plot2_path")

#=============================================================================
Plot 3: Relative difference due to AP effect
=============================================================================#

println("\n[5/5] Generating Plot 3: Relative AP effect...")

# Compute relative differences (in percent)
rel_diff_P0 = @. 100 * (P0_AP - P0) / P0
rel_diff_P2 = @. 100 * (P2_AP - P2) / P2
rel_diff_P4 = @. 100 * (P4_AP - P4) / P4

# Create plot
p3 = plot(k_grid, rel_diff_P0,
    label=L"\ell=0",
    xlabel=L"k\ [h/\mathrm{Mpc}]",
    ylabel=L"\mathrm{Relative\ difference}\ [\%]",
    linewidth=2.5,
    legend=:topright,
    color=1
)
plot!(p3, k_grid, rel_diff_P2,
    label=L"\ell=2",
    linewidth=2.5,
    color=2
)
plot!(p3, k_grid, rel_diff_P4,
    label=L"\ell=4",
    linewidth=2.5,
    color=3
)
hline!(p3, [0], color=:black, linestyle=:dot, label="", linewidth=1.5)
title!(p3, L"\mathrm{Relative\ Impact\ of\ AP\ Effect}")

# Save
plot3_path = joinpath(output_dir, "ap_relative_difference.png")
savefig(p3, plot3_path)
println("   ✓ Saved: $plot3_path")

#=============================================================================
Summary
=============================================================================#

println("\n" * "=" ^ 70)
println("Plot Generation Complete!")
println("=" ^ 70)
println("\nGenerated files:")
println("  1. multipoles_no_ap.png       - Power spectrum multipoles")
println("  2. ap_effect_comparison.png   - Before/after AP corrections")
println("  3. ap_relative_difference.png - Relative AP effect")
println("\nOutput directory: $output_dir")
println("\nNext steps:")
println("  1. Upload these images to cloud storage (Zenodo, GitHub releases, etc.)")
println("  2. Update docs/src/example.md to reference the uploaded URLs")
println("  3. Remove @example blocks that generate plots from example.md")
println("=" ^ 70)
