# Example Usage

This page demonstrates how to use `Effort.jl` to efficiently compute power spectrum multipoles for the Effective Field Theory of Large Scale Structure (EFTofLSS).

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using BenchmarkTools
using Effort
using LaTeXStrings

# Load pre-computed benchmarks using BenchmarkTools native format
benchmark_file = joinpath(@__DIR__, "assets", "effort_benchmark.json")
if isfile(benchmark_file)
    global saved_benchmarks = BenchmarkTools.load(benchmark_file)[1]["Effort"]
else
    @warn "Benchmark file not found at $benchmark_file. Run docs/run_benchmarks.jl first."
    global saved_benchmarks = nothing
end

# Helper function to display benchmark results
function show_benchmark(name)
    if saved_benchmarks === nothing
        return "Benchmarks not available (run docs/run_benchmarks.jl)"
    end

    trial = saved_benchmarks[name]
    time_μs = median(trial).time / 1000
    mem_kb = median(trial).memory / 1024
    allocs = median(trial).allocs

    println("BenchmarkTools.Trial: $(length(trial)) samples with $(trial.params.evals) evaluation(s) per sample.")
    println("  Median time: $(round(time_μs, digits=3)) μs")
    println("  Memory estimate: $(round(mem_kb, digits=2)) KB")
    println("  Allocs estimate: $allocs")
end

# Set LaTeX font for all plots
default(
    fontfamily = "Computer Modern",
    guidefont = (12, :black),
    tickfont = (10, :black),
    legendfont = (10, :black),
    palette = palette(:tab10),
    framestyle = :box,
    grid = false,
    minorticks = true
)
```

## Overview

`Effort.jl` provides a complete, differentiable pipeline for computing galaxy power spectrum multipoles:

1. **Define cosmology** - Set cosmological parameters
2. **Compute growth factors** - Solve ODEs for D(z) and f(z)
3. **Predict multipoles** - Use pre-trained neural network emulators
4. **Apply AP corrections** - Account for Alcock-Paczynski effects (optional)
5. **Window convolution** - Apply survey window functions (optional)

The package ships with pre-trained emulators for the PyBird code, trained on the `mnuw0wacdm` cosmology (9 parameters: redshift ``z``, ``\ln(10^{10}A_{\mathrm{s}})``, ``n_{\mathrm{s}}``, ``H_0``, ``\omega_b``, ``\omega_{\mathrm{cdm}}``, ``\Sigma m_\nu``, ``w_0``, ``w_a``).

---

## Step 1: Load Pre-trained Emulators

`Effort.jl` automatically loads pre-trained emulators during package initialization. The emulators are stored in the `trained_emulators` dictionary:

```@example tutorial
using Effort

# Access the monopole (ℓ=0), quadrupole (ℓ=2), and hexadecapole (ℓ=4) emulators
monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

println("Available emulators: ", keys(Effort.trained_emulators))
println("Monopole emulator loaded successfully!")
nothing # hide
```

Each multipole emulator contains three component emulators (P11, Ploop, Pct) plus bias combination functions. Let's inspect the k-grid:

```@example tutorial
k_grid = vec(monopole_emu.P11.kgrid)
println("k-grid range: [$(minimum(k_grid)), $(maximum(k_grid))] h/Mpc")
println("Number of k-points: $(length(k_grid))")
nothing # hide
```

---

## Step 2: Define Cosmology

Create a cosmology object using the `w0waCDMCosmology` type. This includes standard ΛCDM parameters plus extensions for massive neutrinos, dark energy equation of state, and spatial curvature:

```@example tutorial
# Fiducial Planck-like cosmology (flat universe)
cosmology = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.044,     # Log primordial amplitude: ln(10^10 A_s)
    nₛ = 0.9649,        # Spectral index
    h = 0.6736,         # Reduced Hubble constant: H0 = 100h km/s/Mpc
    ωb = 0.02237,       # Physical baryon density: Ωb h²
    ωc = 0.12,          # Physical cold dark matter density: Ωcdm h²
    mν = 0.06,          # Sum of neutrino masses [eV]
    w0 = -1.0,          # Dark energy EOS at z=0
    wa = 0.0,           # Dark energy EOS evolution parameter
    ωk = 0.0            # Physical curvature density: Ωk h² (default: 0.0 for flat)
)

# Reference cosmology for AP corrections (intentionally different to show AP effect)
cosmo_ref = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.0, nₛ = 0.96, h = 0.70,      # Different h
    ωb = 0.022, ωc = 0.115, mν = 0.06,      # Different ωc
    w0 = -0.95, wa = 0.0, ωk = 0.0          # Different w0 (non-ΛCDM)
)

println("Cosmology defined successfully!")
println("  Flat universe: ωk = $(cosmology.ωk)")
nothing # hide
```

!!! note "Non-flat Universes"
    The `w0waCDMCosmology` type supports non-flat universes through the `ωk` parameter:
    - `ωk = 0.0`: Flat universe (Ωk = 0) - **default**
    - `ωk > 0.0`: Open universe (Ωk > 0, negative spatial curvature)
    - `ωk < 0.0`: Closed universe (Ωk < 0, positive spatial curvature)

    The curvature affects the Hubble parameter E(z) and all distance measures (dA, dL, etc.).

---

## Step 3: Compute Growth Factor and Growth Rate

The growth factor D(z) and growth rate f(z) are computed by solving the differential equation:

```math
D''(a) + \left(2 + \frac{E'(a)}{E(a)}\right)D'(a) = \frac{3}{2}\Omega_m(a)D(a)
```

where E(a) is the normalized Hubble parameter including radiation, matter, massive neutrinos, and dark energy.

```@example tutorial
# Redshift of interest
z = 0.8

# Compute growth factor and growth rate simultaneously
D, f = Effort.D_f_z(z, cosmology)

println("At redshift z = $z:")
println("  Growth factor D(z) = $D")
println("  Growth rate f(z) = $f")
nothing # hide
```

This computation is extremely fast (both D and f computed together in a single ODE solve) and includes full support for automatic differentiation:

```@example tutorial
show_benchmark("D_f_z")
```

The ODE solver accurately accounts for all cosmological components:
- ✓ **Photon radiation** (Ω_γ) - computed from CMB temperature
- ✓ **Cold dark matter + baryons** (Ω_cb) - from ωb and ωc
- ✓ **Massive neutrinos** (Ω_ν) - includes accurate phase-space integrals for energy density
- ✓ **Evolving dark energy** - w(z) = w0 + wa(1-a) parametrization
- ✓ **Spatial curvature** (Ω_k) - supports open, closed, and flat universes via ωk parameter

---

## Step 4: Define Bias Parameters

With "bias parameters" we loosely refer to biases, counterterms, and stochastic contributions. The galaxy power spectrum depends on 11 such parameters in the EFTofLSS framework:

```@example tutorial
# Bias parameters: [b1, b2, b3, b4, cct, cr1, cr2, f, ce0, cemono, cequad]
bias_params = [
    2.0,    # b1
    -0.5,   # b2
    0.3,    # b3
    0.5,    # b4
    0.5,    # cct
    0.5,    # cr1
    0.5,    # cr2
    f,      # f
    1.0,    # ce0
    1.0,    # cemono
    1.0     # cequad
]

println("Bias parameters defined (including f = $f)")
nothing # hide
```

---

## Step 5: Predict Power Spectrum Multipoles

Now we can predict the power spectrum multipoles using the emulators. The emulator expects input in the format: `[z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]`:

```@example tutorial
# Build emulator input array
emulator_input = [
    z,
    cosmology.ln10Aₛ,
    cosmology.nₛ,
    cosmology.h * 100,  # Convert h to H0
    cosmology.ωb,
    cosmology.ωc,
    cosmology.mν,
    cosmology.w0,
    cosmology.wa
]

# Predict monopole, quadrupole, and hexadecapole
P0 = Effort.get_Pℓ(emulator_input, D, bias_params, monopole_emu)
P2 = Effort.get_Pℓ(emulator_input, D, bias_params, quadrupole_emu)
P4 = Effort.get_Pℓ(emulator_input, D, bias_params, hexadecapole_emu)

println("Multipoles computed successfully!")
println("  Monopole P0: $(length(P0)) k-points")
println("  Quadrupole P2: $(length(P2)) k-points")
println("  Hexadecapole P4: $(length(P4)) k-points")
nothing # hide
```

Let's visualize the results:

```@raw html
<img width="800" alt="Power Spectrum Multipoles" src="https://github.com/user-attachments/assets/55d84c98-65ca-429f-b782-52c5da5d6200" />
```

*Figure 1: Power spectrum multipoles at z = 0.8. The monopole (ℓ=0) dominates, with subdominant contributions from the quadrupole (ℓ=2) and hexadecapole (ℓ=4).*

This computation is **extremely fast** - evaluating a single multipole takes only ~28 μs:

```@example tutorial
show_benchmark("monopole")
```

---

## Step 6: Apply Alcock-Paczynski (AP) Corrections

When the assumed reference cosmology differs from the true cosmology, observations are distorted by the Alcock-Paczynski effect. The observed power spectrum is related to the true power spectrum by:

```math
P_{\mathrm{obs}}(k_{\mathrm{obs}}, \mu_{\mathrm{obs}}) = \frac{1}{q_\parallel q_\perp^2} \cdot P_g(k_{\text{true}}, \mu_{\text{true}})
```

where the distortion parameters are:

```math
q_\parallel = \frac{E_{\mathrm{ref}}(z)}{E_{\mathrm{true}}(z)}, \quad q_\perp = \frac{d_{A,\mathrm{true}}(z)}{d_{A,\mathrm{ref}}(z)}
```

### Compute AP parameters

```@example tutorial
# Compute q_parallel and q_perpendicular
q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

println("Alcock-Paczynski parameters:")
println("  q_∥ (parallel) = $q_par")
println("  q_⊥ (perpendicular) = $q_perp")
nothing # hide
```

### Apply AP effect using fast Gauss-Lobatto quadrature

`Effort.jl` implements two methods for applying the AP effect:

1. **`apply_AP`** - Fast Gauss-Lobatto quadrature (recommended)
2. **`apply_AP_check`** - Adaptive QuadGK integration (for validation)

The Gauss-Lobatto method is ~200× faster with negligible accuracy loss (<10⁻¹¹% difference):

```@example tutorial
# Apply AP corrections to all three multipoles
P0_AP, P2_AP, P4_AP = Effort.apply_AP(
    k_grid,          # Input k-grid
    k_grid,          # Output k-grid (can be different)
    P0, P2, P4,      # Input multipoles
    q_par, q_perp,   # AP parameters
    n_GL_points=8    # Number of Gauss-Lobatto points (default: 8)
)

println("AP corrections applied successfully!")
nothing # hide
```

Compare before and after AP corrections:

```@raw html
<img width="800" alt="AP Effect Comparison" src="https://github.com/user-attachments/assets/9840dde2-1387-40c6-a4f8-f18abca1c3a4" />
```

*Figure 2: Effect of Alcock-Paczynski corrections on the monopole and quadrupole. The reference cosmology has h = 0.70 (vs. 0.6736), ωc = 0.115 (vs. 0.12), and w0 = -0.95 (vs. -1.0), producing distortion parameters q_∥ ≈ 0.98 and q_⊥ ≈ 0.99. The AP effect is most pronounced at low k.*

For a clearer view of the AP effect magnitude:

```@raw html
<img width="800" alt="AP Relative Difference" src="https://github.com/user-attachments/assets/34d3cf9b-0585-491f-b621-17ac8ce6e885" />
```

*Figure 3: Relative difference between AP-corrected and uncorrected multipoles. The AP effect can cause percent-level shifts in the power spectrum, particularly important for precision cosmology.*

Performance benchmark:

```@example tutorial
show_benchmark("apply_AP")
```

The AP correction for all three multipoles adds only ~32 μs to the computation!

---

## Complete Pipeline: From Cosmology to Observables

Let's put everything together in a single function that goes from cosmological parameters to AP-corrected multipoles:

```@example tutorial
function compute_multipoles(cosmology, z, bias_params, cosmo_ref=cosmology)
    # Step 1: Compute growth factors simultaneously
    D, f = Effort.D_f_z(z, cosmology)

    # Step 2: Update bias parameters with computed f
    bias_with_f = copy(bias_params)
    bias_with_f[8] = f  # 8th parameter is the growth rate

    # Step 3: Build emulator input
    emulator_input = [
        z, cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h * 100,
        cosmology.ωb, cosmology.ωc, cosmology.mν, cosmology.w0, cosmology.wa
    ]

    # Step 4: Predict multipoles
    monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

    P0 = Effort.get_Pℓ(emulator_input, D, bias_with_f, monopole_emu)
    P2 = Effort.get_Pℓ(emulator_input, D, bias_with_f, quadrupole_emu)
    P4 = Effort.get_Pℓ(emulator_input, D, bias_with_f, hexadecapole_emu)

    # Step 5: Apply AP corrections if reference cosmology differs
    if cosmology !== cosmo_ref
        q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)
        k_grid = vec(monopole_emu.P11.kgrid)
        P0, P2, P4 = Effort.apply_AP(k_grid, k_grid, P0, P2, P4, q_par, q_perp)
    end

    return P0, P2, P4
end

# Test the complete pipeline
P0_full, P2_full, P4_full = compute_multipoles(cosmology, z, bias_params, cosmo_ref)

println("Complete pipeline executed successfully!")
nothing # hide
```

**Performance**: Let's benchmark the complete end-to-end pipeline with AP corrections:

```@example tutorial
show_benchmark("complete_pipeline")
```

This is the **actual measured performance** of the complete function, including all overhead. The total time (~308 μs) includes:
- Growth factors D(z) & f(z): ~169 μs
- Multipole emulation (×3): ~75 μs total
- AP corrections: ~32 μs
- Function call overhead and array operations: ~32 μs

Less than **0.31 milliseconds** for the complete pipeline from cosmological parameters to AP-corrected observables! This is orders of magnitude faster than traditional Boltzmann codes like CLASS or CAMB combined with PyBird.

---

## Differentiation and Jacobians: Two Use Cases

`Effort.jl` provides two complementary approaches for computing derivatives, optimized for different scenarios in cosmological parameter inference.

### Use Case 1: Automatic Differentiation for Gradient-Based Inference

When performing MCMC or maximum likelihood estimation with gradient-based algorithms (e.g., Hamiltonian Monte Carlo, variational inference), you can use **automatic differentiation (AD)** directly through the entire pipeline.

`Effort.jl` is fully compatible with Julia's AD ecosystem:
- **ForwardDiff.jl**: Forward-mode AD for efficient gradients
- **Zygote.jl**: Reverse-mode AD for large parameter spaces

This works seamlessly because the package includes:
- **Custom ChainRules**: Hand-written adjoints for critical operations (Akima interpolation, window convolution)
- **SciMLSensitivity**: Efficient gradients through ODE solvers (growth factors)
- **Non-mutating operations**: All functions are Zygote-compatible

**Example - Differentiating the complete pipeline:**

```@example tutorial
using ForwardDiff

# Define a loss function over ALL parameters (cosmological + bias)
function full_pipeline_loss(all_params)
    # Unpack: first 8 are cosmological, next 11 are bias parameters
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params[1], nₛ = all_params[2], h = all_params[3],
        ωb = all_params[4], ωc = all_params[5], mν = all_params[6],
        w0 = all_params[7], wa = all_params[8], ωk = 0.0
    )

    # Run complete pipeline: ODE solve → emulator → power spectrum
    D_local, f_local = Effort.D_f_z(z, cosmo_local)

    # Bias parameters (9-19) with f replaced at index 8
    bias_local = [all_params[9:16]..., f_local, all_params[17:19]...]

    emulator_input_local = [
        z, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h * 100,
        cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa
    ]

    P0_local = Effort.get_Pℓ(emulator_input_local, D_local, bias_local, monopole_emu)
    return sum(abs2, P0_local)  # L2 norm
end

# Pack ALL parameters (8 cosmological + 11 bias = 19 total)
all_params = vcat(
    [cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h,
     cosmology.ωb, cosmology.ωc, cosmology.mν,
     cosmology.w0, cosmology.wa],
    bias_params
)

# Compute gradient using ForwardDiff (19 parameters: 8 cosmo + 11 bias)
grad_all = ForwardDiff.gradient(full_pipeline_loss, all_params)
println("Gradient via ForwardDiff (w.r.t. all 19 parameters):")
println("  ∂L/∂h = $(grad_all[3])")
println("  ∂L/∂ωc = $(grad_all[5])")
println("  ∂L/∂b1 = $(grad_all[9])")
println("  All gradients finite: $(all(isfinite, grad_all))")
nothing # hide
```

**Performance - ForwardDiff (19 parameters):**

```@example tutorial
show_benchmark("forwarddiff_gradient")
```

You can also use Zygote for reverse-mode AD:

```@example tutorial
using Zygote

# Compute gradient using Zygote
grad_zygote = Zygote.gradient(full_pipeline_loss, all_params)[1]
println("Gradient via Zygote (w.r.t. all 19 parameters):")
println("  ∂L/∂h = $(grad_zygote[3])")
println("  ∂L/∂ωc = $(grad_zygote[5])")
println("  ∂L/∂b1 = $(grad_zygote[9])")
println("  All gradients finite: $(all(isfinite, grad_zygote))")
nothing # hide
```

**Performance - Zygote (19 parameters):**

```@example tutorial
show_benchmark("zygote_gradient")
```

**What do these timings mean?**

These benchmarks show the time to compute **all 19 gradients** (∂L/∂ln10Aₛ, ∂L/∂nₛ, ∂L/∂h, ∂L/∂ωb, ∂L/∂ωc, ∂L/∂mν, ∂L/∂w0, ∂L/∂wa, plus all 11 bias parameter gradients) in a single call. The gradient computation includes:
- Differentiating through the ODE solver (growth factors D and f)
- Differentiating through the neural network emulator
- Differentiating through the bias expansion

**ForwardDiff** (~1 ms) is generally faster for problems with fewer parameters, while **Zygote** (~2 ms) is more memory-efficient for large-scale problems. Both are fast enough for gradient-based MCMC sampling.

Both ForwardDiff and Zygote are explicitly tested to ensure reliability. This enables:
- **Hamiltonian Monte Carlo (HMC)** for efficient MCMC sampling
- **Variational Inference (VI)** for fast posterior approximation
- **Gradient-based optimization** for maximum likelihood estimation

### Use Case 2: Analytical Jacobians for Fisher Information Matrices

When computing **Fisher Information Matrices** (needed for survey forecasts, Jeffreys priors, or error propagation), we need Jacobians of the power spectrum with respect to bias parameters.

While these Jacobians *could* be computed with AD, doing so during an MCMC analysis would require **AD over AD** (differentiating the Jacobian computation itself to get likelihood gradients). This is inefficient and numerically unstable.

Instead, `Effort.jl` provides **analytical Jacobian implementations** optimized for this use case:

```@example tutorial
# Compute power spectrum AND its Jacobian w.r.t. bias parameters
P0_jac, J0 = Effort.get_Pℓ_jacobian(emulator_input, D, bias_params, monopole_emu)

println("Analytical Jacobian computed!")
println("  Shape: $(size(J0)) (k-points × bias parameters)")
println("  P0 from get_Pℓ_jacobian matches get_Pℓ: $(P0 ≈ P0_jac)")
nothing # hide
```

These analytical Jacobians can also be AP-corrected efficiently:

```@example tutorial
# Compute Jacobians for all three multipoles
_, J0 = Effort.get_Pℓ_jacobian(emulator_input, D, bias_params, monopole_emu)
_, J2 = Effort.get_Pℓ_jacobian(emulator_input, D, bias_params, quadrupole_emu)
_, J4 = Effort.get_Pℓ_jacobian(emulator_input, D, bias_params, hexadecapole_emu)

# Apply AP corrections (matrix version is optimized for multiple columns)
J0_AP, J2_AP, J4_AP = Effort.apply_AP(k_grid, k_grid, J0, J2, J4, q_par, q_perp)

println("AP-corrected Jacobians computed!")
println("  Each Jacobian shape: $(size(J0_AP))")
nothing # hide
```

**Reliability**: These analytical Jacobians are tested against:
- **Computer Algebra Systems (CAS)**: Symbolic differentiation for exact reference
- **Automatic Differentiation**: Numerical validation with ForwardDiff/Zygote

This ensures correctness while maintaining performance, making them ideal for:
- **Fisher matrix forecasts** for survey optimization
- **Jeffreys priors** computation in Bayesian analyses
- **Efficient MCMC** when combined with AD for cosmological parameters

### Example: Full Pipeline Differentiation

You can also differentiate through the complete pipeline (ODE solvers + emulators + AP corrections) with respect to cosmological parameters:

```julia
using ForwardDiff, Zygote

function full_loss(cosmo_params)
    # Unpack: [ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
    ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params

    cosmo = Effort.w0waCDMCosmology(
        ln10Aₛ=ln10As, nₛ=ns, h=H0/100, ωb=ωb, ωc=ωcdm,
        mν=mν, w0=w0, wa=wa, ωk=0.0
    )

    # Full pipeline: ODE → Emulator → AP
    P0, P2, P4 = compute_multipoles(cosmo, z, bias_params, cosmo_ref)

    return sum(abs2, P0) + sum(abs2, P2) + sum(abs2, P4)
end

# Both ForwardDiff and Zygote work!
grad_fd = ForwardDiff.gradient(full_loss, cosmo_params)
grad_zy = Zygote.gradient(full_loss, cosmo_params)[1]
```

See the test suite in `test/test_pipeline.jl` for complete working examples with all AD backends (ForwardDiff, Zygote, FiniteDifferences).

---

## Multi-Redshift Analysis

Real cosmological analyses often require computing power spectra at multiple redshifts simultaneously. `Effort.jl` efficiently handles this by solving the growth ODE only once for all redshifts.

**Example: 5 redshifts from z=0.8 to z=1.9**

```@example tutorial
# Define multiple redshifts
z_array = range(0.8, 1.9, length=5)
println("Analyzing $(length(z_array)) redshifts: $(collect(z_array))")
nothing # hide
```

The key advantage is that **`D_f_z` accepts vector inputs**, solving the ODE once and evaluating at all redshifts:

```@example tutorial
# Compute growth factors for ALL redshifts at once (single ODE solve!)
D_array, f_array = Effort.D_f_z(z_array, cosmology)

println("Growth factors computed for all redshifts:")
for (i, z_i) in enumerate(z_array)
    println("  z = $(round(z_i, digits=2)): D = $(round(D_array[i], digits=4)), f = $(round(f_array[i], digits=4))")
end
nothing # hide
```

### Multi-Redshift Forward Pass

Here's a complete multi-redshift pipeline that computes power spectra at all 5 redshifts:

```@example tutorial
function multi_z_pipeline(all_params_multi)
    # Unpack: first 8 are cosmological, next 55 are bias (11 × 5 redshifts)
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params_multi[1], nₛ = all_params_multi[2], h = all_params_multi[3],
        ωb = all_params_multi[4], ωc = all_params_multi[5], mν = all_params_multi[6],
        w0 = all_params_multi[7], wa = all_params_multi[8], ωk = 0.0
    )

    # Compute D and f for ALL redshifts at once (single ODE solve!)
    D_array, f_array = Effort.D_f_z(z_array, cosmo_local)

    # Compute power spectra for all redshifts
    total_loss = 0.0
    for (i, z_i) in enumerate(z_array)
        # Bias parameters for this redshift
        bias_start = 8 + (i-1)*11 + 1
        bias_end = 8 + i*11
        bias_this_z = [all_params_multi[bias_start:bias_start+6]...,
                       f_array[i],
                       all_params_multi[bias_start+7:bias_end]...]

        emulator_input_local = [
            z_i, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h * 100,
            cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa
        ]

        P0_local = Effort.get_Pℓ(emulator_input_local, D_array[i], bias_this_z, monopole_emu)
        total_loss += sum(abs2, P0_local)
    end

    return total_loss
end

# Parameters: 8 cosmo + 55 bias (11 × 5 redshifts) = 63 total
all_params_multi = vcat(
    [cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h,
     cosmology.ωb, cosmology.ωc, cosmology.mν,
     cosmology.w0, cosmology.wa],
    repeat(bias_params, 5)
)

result = multi_z_pipeline(all_params_multi)
println("Multi-redshift pipeline executed successfully!")
println("Total parameters: $(length(all_params_multi)) (8 cosmo + 55 bias)")
nothing # hide
```

**Performance:**

```@example tutorial
show_benchmark("multi_z_forward")
```

Computing power spectra for **5 redshifts** takes only ~368 μs - barely more than a single redshift (~313 μs)! This is because the expensive ODE solve is done only once.

### Multi-Redshift Differentiation

The multi-redshift pipeline is fully differentiable with both ForwardDiff and Zygote:

```@example tutorial
using ForwardDiff, Zygote

# ForwardDiff: all 63 gradients
grad_fd = ForwardDiff.gradient(multi_z_pipeline, all_params_multi)
println("ForwardDiff gradient computed!")
println("  Gradient shape: $(length(grad_fd)) (8 cosmo + 55 bias)")
println("  ∂L/∂h = $(grad_fd[3])")
println("  All gradients finite: $(all(isfinite, grad_fd))")
nothing # hide
```

**Performance - ForwardDiff (63 parameters):**

```@example tutorial
show_benchmark("multi_z_forwarddiff")
```

```@example tutorial
# Zygote: all 63 gradients
grad_zy = Zygote.gradient(multi_z_pipeline, all_params_multi)[1]
println("Zygote gradient computed!")
println("  Gradient shape: $(length(grad_zy)) (8 cosmo + 55 bias)")
println("  ∂L/∂h = $(grad_zy[3])")
println("  All gradients finite: $(all(isfinite, grad_zy))")
nothing # hide
```

**Performance - Zygote (63 parameters):**

```@example tutorial
show_benchmark("multi_z_zygote")
```

**Key Observations:**
- **Zygote** (~6 ms) is **2× faster** than ForwardDiff (~13 ms) for 63 parameters
- Zygote's reverse-mode AD becomes more efficient as parameter count increases
- Both are fast enough for multi-redshift MCMC analyses

---

## Performance Summary

Here's a summary of computational timings for key operations:

| Operation | Time | Memory | Allocs | Speedup vs PyBird |
|-----------|------|--------|--------|-------------------|
| Growth factors D(z) & f(z) | 183 μs | 276 KB | 11,805 | ~1000× |
| Single multipole (ℓ=0) | 25 μs | 92 KB | 186 | ~10,000× |
| Single multipole (ℓ=2) | 25 μs | 93 KB | 188 | ~10,000× |
| Single multipole (ℓ=4) | 25 μs | 90 KB | 180 | ~10,000× |
| AP correction (3 multipoles) | 32 μs | 86 KB | 208 | ~100× |
| **Complete pipeline (1z)** | **313 μs** | **650 KB** | **12,961** | **~10,000×** |
| ForwardDiff gradient (19 params) | 1.06 ms | 4.30 MB | 21,095 | - |
| Zygote gradient (19 params) | 1.97 ms | 2.91 MB | 42,803 | - |
| **Multi-z forward (5z, 63 params)** | **368 μs** | **744 KB** | **12,921** | **~9,000×** |
| Multi-z ForwardDiff (63 params) | 12.70 ms | 39.80 MB | 69,630 | - |
| Multi-z Zygote (63 params) | 6.04 ms | 18.98 MB | 59,243 | - |

```@example tutorial
# Display system information for reproducibility
using JSON
metadata_file = joinpath(@__DIR__, "assets", "benchmark_metadata.json")
if isfile(metadata_file)
    metadata = JSON.parsefile(metadata_file)
    println("Benchmark Hardware Information:")
    println("  Julia version: ", get(metadata, "julia_version", "N/A"))
    println("  CPU: ", get(metadata, "cpu_info", "N/A"))
    println("  Cores: ", get(metadata, "ncores", "N/A"))
    println("  Timestamp: ", get(metadata, "timestamp", "N/A"))
else
    println("System Information:")
    println("  Julia version: ", VERSION)
    println("  CPU: ", Sys.cpu_info()[1].model)
    println("  Cores: ", Sys.CPU_THREADS)
end
```

!!! note "Benchmark Details"
    These benchmarks were run locally and saved to avoid recomputing during CI/CD. To regenerate on your hardware:
    ```bash
    julia --project=docs docs/run_benchmarks.jl
    ```
    The script will display your system information and save detailed results (min/max/mean statistics) to `docs/src/assets/effort_benchmark.json`.

The entire pipeline is **differentiable** and **extremely fast**, making it ideal for:
- Large-scale MCMC analysis (DESI, Euclid, LSST)
- Real-time parameter inference
- Fisher forecasts and survey optimization
- Gradient-based inference methods (HMC, VI)

---

## Summary

This tutorial demonstrated the complete workflow for using `Effort.jl`:

1. ✅ Load pre-trained emulators from artifacts
2. ✅ Define cosmological parameters
3. ✅ Compute growth factors by solving ODEs
4. ✅ Predict power spectrum multipoles using neural networks
5. ✅ Apply Alcock-Paczynski corrections
6. ✅ Compute Jacobians for efficient inference
7. ✅ Differentiate through the entire pipeline with AD

The package achieves **~10,000× speedup** compared to traditional codes while maintaining full differentiability, enabling next-generation inference methods for cosmological surveys.

For more details on the API, see the API Documentation pages in the sidebar.
