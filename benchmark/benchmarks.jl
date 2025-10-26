using BenchmarkTools
using Effort
using AbstractCosmologicalEmulators
using LinearAlgebra
using DataInterpolations
using LegendrePolynomials
using Artifacts

# Load extension dependencies for Background cosmology benchmarks
using OrdinaryDiffEqTsit5
using Integrals
using FastGaussQuadrature
using Zygote

# Every benchmark file must define a BenchmarkGroup named SUITE.
const SUITE = BenchmarkGroup()


# Store references to emulators for benchmarking
const emulator_0 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
const emulator_2 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
const emulator_4 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

# Define test input sizes and parameters
const nk_test = 100  # number of k points
const n_eft_params = 9  # number of EFT parameters (for neural network components)

# Create test data
const k_test = collect(range(0.01, stop=0.3, length=nk_test))
const eft_params_test = randn(n_eft_params)

# --- Benchmark Groups ---
SUITE["emulator"] = BenchmarkGroup(["multipoles", "neural_network"])
SUITE["projection"] = BenchmarkGroup(["legendre", "AP_effect"])
SUITE["integration"] = BenchmarkGroup(["window_convolution"])
SUITE["background"] = BenchmarkGroup(["cosmology"])

# --- Multipole Emulator Benchmarks ---
# We benchmark the neural network component directly
# Each emulator has P11, Ploop, and Pct components with trained neural networks

# Benchmark P11 component of monopole emulator
SUITE["emulator"]["monopole_P11"] = @benchmarkable begin
    Effort.get_component(params, D, $emulator_0.P11)
end setup = (
    params = copy($eft_params_test);
    D = 1.0  # Growth factor (typical value)
)

# Benchmark the raw neural network evaluation (most direct benchmark)
SUITE["emulator"]["raw_nn_monopole"] = @benchmarkable begin
    norm_input = AbstractCosmologicalEmulators.maximin(params, $emulator_0.P11.InMinMax)
    norm_output = AbstractCosmologicalEmulators.run_emulator(norm_input, $emulator_0.P11.TrainedEmulator)
    norm_output
end setup = (
    params = copy($eft_params_test)
)

# --- Legendre Polynomial Benchmarks ---
SUITE["projection"]["legendre_0"] = @benchmarkable Effort._Legendre_0(μ) setup = (
    μ = rand()
)

SUITE["projection"]["legendre_2"] = @benchmarkable Effort._Legendre_2(μ) setup = (
    μ = rand()
)

SUITE["projection"]["legendre_4"] = @benchmarkable Effort._Legendre_4(μ) setup = (
    μ = rand()
)

SUITE["projection"]["legendre_array"] = @benchmarkable begin
    Effort._Legendre_0.(μ_vals)
    Effort._Legendre_2.(μ_vals)
    Effort._Legendre_4.(μ_vals)
end setup = (
    μ_vals = rand(100)
)

# --- Akima Interpolation Benchmarks ---
# Matrix Akima optimization provides ~2.4x speedup for Jacobian operations
# by computing shared operations (diff(t), interval finding) once instead of per-column
SUITE["interpolation"] = BenchmarkGroup(["akima", "scalar", "matrix"])

# Benchmark scalar (vector) Akima interpolation
SUITE["interpolation"]["akima_scalar"] = @benchmarkable begin
    Effort._akima_interpolation(u, t, t_new)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50)
)

# Benchmark optimized matrix Akima interpolation (11 columns - typical Jacobian size)
# Uses matrix-native implementation with shared diff(t) computation
# Expected: ~2.4x faster than naive column-wise approach
SUITE["interpolation"]["akima_matrix_11cols_optimized"] = @benchmarkable begin
    Effort._akima_interpolation(u, t, t_new)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 11)
)

# Benchmark naive column-by-column Akima (for comparison with optimized version)
# This represents the old approach before matrix optimization
# Expected: ~2.4x slower than optimized matrix version
SUITE["interpolation"]["akima_matrix_11cols_naive"] = @benchmarkable begin
    hcat([Effort._akima_interpolation(u[:, i], t, t_new) for i in 1:11]...)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 11)
)

# Benchmark with larger matrix (20 columns) to test scalability
# Speedup should be even better with more columns
SUITE["interpolation"]["akima_matrix_20cols_optimized"] = @benchmarkable begin
    Effort._akima_interpolation(u, t, t_new)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 20)
)

SUITE["interpolation"]["akima_matrix_20cols_naive"] = @benchmarkable begin
    hcat([Effort._akima_interpolation(u[:, i], t, t_new) for i in 1:20]...)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 20)
)

# Benchmark with smaller matrix (3 columns) - minimum realistic case
SUITE["interpolation"]["akima_matrix_3cols_optimized"] = @benchmarkable begin
    Effort._akima_interpolation(u, t, t_new)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 3)
)

SUITE["interpolation"]["akima_matrix_3cols_naive"] = @benchmarkable begin
    hcat([Effort._akima_interpolation(u[:, i], t, t_new) for i in 1:3]...)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 3)
)

# --- Akima Interpolation with Automatic Differentiation ---
# Benchmark gradients through matrix Akima (critical for training)
SUITE["interpolation"]["akima_gradient_zygote"] = @benchmarkable begin
    Zygote.gradient(u_mat -> sum(Effort._akima_interpolation(u_mat, t, t_new)), u)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 11)
)

# Benchmark gradient w.r.t. output grid (used in AP transformations)
SUITE["interpolation"]["akima_gradient_tnew_zygote"] = @benchmarkable begin
    Zygote.gradient(tn -> sum(Effort._akima_interpolation(u, t, tn)), t_new)
end setup = (
    t = collect(range(0.01, 0.3, length=50));
    t_new = collect(range(0.015, 0.28, length=100));
    u = randn(50, 11)
)

# --- AP Effect Benchmarks ---
# Create test multipole arrays
const mono_test = randn(nk_test)
const quad_test = randn(nk_test)
const hexa_test = randn(nk_test)

SUITE["projection"]["apply_AP"] = @benchmarkable begin
    Effort.apply_AP(k_input, k_output, mono, quad, hexa, q_par, q_perp)
end setup = (
    k_input = copy($k_test);
    k_output = collect(range(0.015, stop=0.25, length=80));
    mono = copy($mono_test);
    quad = copy($quad_test);
    hexa = copy($hexa_test);
    q_par = 1.02;
    q_perp = 0.98
)

# Benchmark the check version (slower, uses numerical integration)
SUITE["projection"]["apply_AP_check"] = @benchmarkable begin
    Effort.apply_AP_check(k_input, k_output, mono, quad, hexa, q_par, q_perp)
end setup = (
    k_input = collect(range(0.01, stop=0.1, length=20));  # Smaller for check version
    k_output = collect(range(0.015, stop=0.09, length=10));
    mono = randn(20);
    quad = randn(20);
    hexa = randn(20);
    q_par = 1.02;
    q_perp = 0.98
)

# --- Window Convolution Benchmarks ---
SUITE["integration"]["window_4D"] = @benchmarkable begin
    Effort.window_convolution(W, v)
end setup = (
    W = randn(10, 20, 10, 30);
    v = randn(20, 30)
)

SUITE["integration"]["window_2D"] = @benchmarkable begin
    Effort.window_convolution(W, v)
end setup = (
    W = randn(50, 100);
    v = randn(100)
)

# --- Background Cosmology Benchmarks (if extension is loaded) ---
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

if !isnothing(ext)
    # Create test cosmologies for benchmarking
    const cosmo_mcmc = Effort.w0waCDMCosmology(
        h = 0.7,
        ωb = 0.022,
        ωc = 0.12,
        mν = 0.06,
        w0 = -0.95,
        wa = 0.1
    )

    const cosmo_ref = Effort.w0waCDMCosmology(
        h = 0.67,
        ωb = 0.022,
        ωc = 0.12,
        mν = 0.06,
        w0 = -1.0,
        wa = 0.0
    )

    SUITE["background"]["E_z"] = @benchmarkable Effort.E_z(z, $cosmo_mcmc) setup = (
        z = 1.0
    )

    SUITE["background"]["D_z"] = @benchmarkable Effort.D_z(z, $cosmo_mcmc) setup = (
        z = 0.5
    )

    SUITE["background"]["f_z"] = @benchmarkable Effort.f_z(z, $cosmo_mcmc) setup = (
        z = 0.5
    )

    SUITE["background"]["D_f_z"] = @benchmarkable Effort.D_f_z(z, $cosmo_mcmc) setup = (
        z = 0.5
    )

    SUITE["background"]["q_par_perp"] = @benchmarkable begin
        Effort.q_par_perp(z, $cosmo_mcmc, $cosmo_ref)
    end setup = (
        z = 0.5
    )
end

# --- Gradient Benchmarks (using Zygote) ---
SUITE["gradients"] = BenchmarkGroup(["zygote"])

# Benchmark gradient of component evaluation
SUITE["gradients"]["component_gradient"] = @benchmarkable begin
    Zygote.gradient(params) do p
        result = Effort.get_component(p, D, $emulator_0.P11)
        sum(result)  # Need a scalar output for gradient
    end
end setup = (
    params = randn($n_eft_params);
    D = 1.0
)

# --- Jacobian Benchmarks ---
SUITE["jacobian"] = BenchmarkGroup(["analytical", "AP_batch"])

# Create test cosmology and bias parameters for Jacobian benchmarks
const cosmology_jac_test = [0.8, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
const bias_jac_test = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
const D_jac_test = 0.75

# AP parameters for Jacobian benchmarks
const cosmo_jac_mcmc = Effort.w0waCDMCosmology(ln10Aₛ=3.044, nₛ=0.9649, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
const cosmo_jac_ref = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.022, ωc=0.119, mν=0.06, w0=-1.0, wa=0.0)
const qpar_jac, qperp_jac = Effort.q_par_perp(0.8, cosmo_jac_mcmc, cosmo_jac_ref)
const k_grid_jac = vec(emulator_0.P11.kgrid)

# Benchmark analytical Jacobian computation (without AP)
SUITE["jacobian"]["get_Pℓ_jacobian"] = @benchmarkable begin
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_0)
end setup = (
    cosmology = copy($cosmology_jac_test);
    D = $D_jac_test;
    bias = copy($bias_jac_test)
)

# Benchmark all three multipoles Jacobian computation
SUITE["jacobian"]["get_all_Pℓ_jacobians"] = @benchmarkable begin
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_0)
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_2)
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_4)
end setup = (
    cosmology = copy($cosmology_jac_test);
    D = $D_jac_test;
    bias = copy($bias_jac_test)
)

# Benchmark optimized batch apply_AP on Jacobian matrices
# Uses matrix Akima interpolation for ~3x speedup over naive column-wise approach
SUITE["jacobian"]["apply_AP_batch"] = @benchmarkable begin
    Effort.apply_AP(k_input, k_output, Jac0, Jac2, Jac4, q_par, q_perp, n_GL_points=8)
end setup = (
    k_input = $k_grid_jac;
    k_output = $k_grid_jac;
    result0 = Effort.get_Pℓ_jacobian($cosmology_jac_test, $D_jac_test, $bias_jac_test, $emulator_0);
    Jac0 = result0[2];
    result2 = Effort.get_Pℓ_jacobian($cosmology_jac_test, $D_jac_test, $bias_jac_test, $emulator_2);
    Jac2 = result2[2];
    result4 = Effort.get_Pℓ_jacobian($cosmology_jac_test, $D_jac_test, $bias_jac_test, $emulator_4);
    Jac4 = result4[2];
    q_par = $qpar_jac;
    q_perp = $qperp_jac
)

# Benchmark naive column-wise apply_AP (for comparison with optimized batch version)
# This represents the old approach before matrix Akima optimization
SUITE["jacobian"]["apply_AP_columnwise"] = @benchmarkable begin
    n_params = size(Jac0, 2)
    for col in 1:n_params
        Effort.apply_AP(k_input, k_output, Jac0[:, col], Jac2[:, col], Jac4[:, col],
                       q_par, q_perp, n_GL_points=8)
    end
end setup = (
    k_input = $k_grid_jac;
    k_output = $k_grid_jac;
    result0 = Effort.get_Pℓ_jacobian($cosmology_jac_test, $D_jac_test, $bias_jac_test, $emulator_0);
    Jac0 = result0[2];
    result2 = Effort.get_Pℓ_jacobian($cosmology_jac_test, $D_jac_test, $bias_jac_test, $emulator_2);
    Jac2 = result2[2];
    result4 = Effort.get_Pℓ_jacobian($cosmology_jac_test, $D_jac_test, $bias_jac_test, $emulator_4);
    Jac4 = result4[2];
    q_par = $qpar_jac;
    q_perp = $qperp_jac
)

# Benchmark full pipeline: Jacobian computation (without AP to avoid dispatch issues)
SUITE["jacobian"]["full_jacobian_computation"] = @benchmarkable begin
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_0)
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_2)
    Effort.get_Pℓ_jacobian(cosmology, D, bias, $emulator_4)
end setup = (
    cosmology = copy($cosmology_jac_test);
    D = $D_jac_test;
    bias = copy($bias_jac_test)
)

# --- Full Pipeline Differentiation Benchmarks (ODE → Emulator → AP) ---
SUITE["full_pipeline"] = BenchmarkGroup(["gradients", "ODE", "emulator", "AP"])

# Test parameters for full pipeline benchmarks
const z_pipeline = 0.8
const cosmo_params_pipeline = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
const bias_params_pipeline = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
const cosmo_ref_pipeline = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.0, nₛ = 0.96, h = 0.67,
    ωb = 0.022, ωc = 0.119, mν = 0.06,
    w0 = -1.0, wa = 0.0
)
const k_grid_pipeline = vec(emulator_0.P11.kgrid)

# Complete pipeline function (used in both benchmarks)
function complete_pipeline_bench(cosmo_params_vector)
    ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
    h = H0 / 100.0

    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ = ln10As, nₛ = ns, h = h,
        ωb = ωb, ωc = ωcdm, mν = mν,
        w0 = w0, wa = wa
    )

    # Compute D and f simultaneously (more efficient than separate calls)
    D, f = Effort.D_f_z(z_pipeline, cosmology)

    bias_with_f = [bias_params_pipeline[1], bias_params_pipeline[2], bias_params_pipeline[3],
                  bias_params_pipeline[4], bias_params_pipeline[5], bias_params_pipeline[6],
                  bias_params_pipeline[7], f, bias_params_pipeline[9],
                  bias_params_pipeline[10], bias_params_pipeline[11]]

    emulator_params = [z_pipeline, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

    P0 = Effort.get_Pℓ(emulator_params, D, bias_with_f, emulator_0)
    P2 = Effort.get_Pℓ(emulator_params, D, bias_with_f, emulator_2)
    P4 = Effort.get_Pℓ(emulator_params, D, bias_with_f, emulator_4)

    q_par, q_perp = Effort.q_par_perp(z_pipeline, cosmology, cosmo_ref_pipeline)

    P0_AP, P2_AP, P4_AP = Effort.apply_AP(
        k_grid_pipeline, k_grid_pipeline, P0, P2, P4,
        q_par, q_perp, n_GL_points=8
    )

    return sum(P0_AP) + sum(P2_AP) + sum(P4_AP)
end

# Complete pipeline with Jacobians function
function complete_pipeline_jacobians_bench(cosmo_params_vector)
    ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
    h = H0 / 100.0

    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ = ln10As, nₛ = ns, h = h,
        ωb = ωb, ωc = ωcdm, mν = mν,
        w0 = w0, wa = wa
    )

    # Compute D and f simultaneously (more efficient than separate calls)
    D, f = Effort.D_f_z(z_pipeline, cosmology)

    bias_with_f = [bias_params_pipeline[1], bias_params_pipeline[2], bias_params_pipeline[3],
                  bias_params_pipeline[4], bias_params_pipeline[5], bias_params_pipeline[6],
                  bias_params_pipeline[7], f, bias_params_pipeline[9],
                  bias_params_pipeline[10], bias_params_pipeline[11]]

    emulator_params = [z_pipeline, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

    _, Jac0 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, emulator_0)
    _, Jac2 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, emulator_2)
    _, Jac4 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, emulator_4)

    q_par, q_perp = Effort.q_par_perp(z_pipeline, cosmology, cosmo_ref_pipeline)

    Jac0_AP, Jac2_AP, Jac4_AP = Effort.apply_AP(
        k_grid_pipeline, k_grid_pipeline, Jac0, Jac2, Jac4,
        q_par, q_perp, n_GL_points=8
    )

    return sum(Jac0_AP) + sum(Jac2_AP) + sum(Jac4_AP)
end

# Benchmark: Forward pass of complete pipeline (ODE → Emulator → AP)
SUITE["full_pipeline"]["forward_pass"] = @benchmarkable begin
    complete_pipeline_bench(cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Benchmark: ForwardDiff gradient through complete pipeline
# This works with both power spectra and demonstrates full differentiability
using ForwardDiff
SUITE["full_pipeline"]["gradient_forwarddiff"] = @benchmarkable begin
    ForwardDiff.gradient(complete_pipeline_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Benchmark: Zygote gradient through complete pipeline
# This works with power spectra (Zygote has issues with Jacobian internals)
SUITE["full_pipeline"]["gradient_zygote"] = @benchmarkable begin
    Zygote.gradient(complete_pipeline_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Benchmark: Forward pass with Jacobians (ODE → Jacobian → AP)
SUITE["full_pipeline"]["forward_pass_jacobians"] = @benchmarkable begin
    complete_pipeline_jacobians_bench(cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Benchmark: ForwardDiff gradient through Jacobian pipeline
SUITE["full_pipeline"]["gradient_forwarddiff_jacobians"] = @benchmarkable begin
    ForwardDiff.gradient(complete_pipeline_jacobians_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Benchmark: Zygote gradient through Jacobian pipeline (now works!)
SUITE["full_pipeline"]["gradient_zygote_jacobians"] = @benchmarkable begin
    Zygote.gradient(complete_pipeline_jacobians_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

#=============================================================================#
# Multi-Redshift Pipeline Benchmarks
#=============================================================================#

# Multi-redshift setup: 5 redshift bins
const z_bins_multiz = range(0.9, 1.8, length=5) |> collect

# Multi-redshift Pℓ + AP pipeline (8 cosmological parameters)
# Uses vectorized D_f_z to compute growth factors for all redshifts at once (single ODE solve!)
function multiz_pl_ap_bench(cosmo_params_vector)
    ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
    h = H0 / 100.0

    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ = ln10As, nₛ = ns, h = h,
        ωb = ωb, ωc = ωcdm, mν = mν,
        w0 = w0, wa = wa
    )

    # Compute D and f for ALL redshifts at once (single ODE solve!)
    D_array, f_array = Effort.D_f_z(z_bins_multiz, cosmology)

    total = 0.0
    for (i, z) in enumerate(z_bins_multiz)
        bias_with_f = [bias_params_pipeline[1], bias_params_pipeline[2], bias_params_pipeline[3],
                      bias_params_pipeline[4], bias_params_pipeline[5], bias_params_pipeline[6],
                      bias_params_pipeline[7], f_array[i], bias_params_pipeline[9],
                      bias_params_pipeline[10], bias_params_pipeline[11]]

        emulator_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

        P0 = Effort.get_Pℓ(emulator_params, D_array[i], bias_with_f, emulator_0)
        P2 = Effort.get_Pℓ(emulator_params, D_array[i], bias_with_f, emulator_2)
        P4 = Effort.get_Pℓ(emulator_params, D_array[i], bias_with_f, emulator_4)

        q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref_pipeline)

        P0_AP, P2_AP, P4_AP = Effort.apply_AP(
            k_grid_pipeline, k_grid_pipeline, P0, P2, P4,
            q_par, q_perp, n_GL_points=8
        )

        total += sum(P0_AP) + sum(P2_AP) + sum(P4_AP)
    end

    return total
end

# Multi-redshift Jacobian + AP pipeline (8 cosmological parameters)
# Uses vectorized D_f_z to compute growth factors for all redshifts at once (single ODE solve!)
function multiz_jac_ap_bench(cosmo_params_vector)
    ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
    h = H0 / 100.0

    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ = ln10As, nₛ = ns, h = h,
        ωb = ωb, ωc = ωcdm, mν = mν,
        w0 = w0, wa = wa
    )

    # Compute D and f for ALL redshifts at once (single ODE solve!)
    D_array, f_array = Effort.D_f_z(z_bins_multiz, cosmology)

    total = 0.0
    for (i, z) in enumerate(z_bins_multiz)
        bias_with_f = [bias_params_pipeline[1], bias_params_pipeline[2], bias_params_pipeline[3],
                      bias_params_pipeline[4], bias_params_pipeline[5], bias_params_pipeline[6],
                      bias_params_pipeline[7], f_array[i], bias_params_pipeline[9],
                      bias_params_pipeline[10], bias_params_pipeline[11]]

        emulator_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

        _, Jac0 = Effort.get_Pℓ_jacobian(emulator_params, D_array[i], bias_with_f, emulator_0)
        _, Jac2 = Effort.get_Pℓ_jacobian(emulator_params, D_array[i], bias_with_f, emulator_2)
        _, Jac4 = Effort.get_Pℓ_jacobian(emulator_params, D_array[i], bias_with_f, emulator_4)

        q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref_pipeline)

        Jac0_AP, Jac2_AP, Jac4_AP = Effort.apply_AP(
            k_grid_pipeline, k_grid_pipeline, Jac0, Jac2, Jac4,
            q_par, q_perp, n_GL_points=8
        )

        total += sum(Jac0_AP) + sum(Jac2_AP) + sum(Jac4_AP)
    end

    return total
end

# Multi-z pipeline benchmarks
SUITE["multiz_pipeline"] = BenchmarkGroup(["multi_redshift", "gradients", "ODE", "emulator", "AP"])

# Forward passes
SUITE["multiz_pipeline"]["forward_multiz_pl_ap"] = @benchmarkable begin
    multiz_pl_ap_bench(cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

SUITE["multiz_pipeline"]["forward_multiz_jac_ap"] = @benchmarkable begin
    multiz_jac_ap_bench(cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# ForwardDiff gradients
SUITE["multiz_pipeline"]["gradient_forwarddiff_multiz_pl_ap"] = @benchmarkable begin
    ForwardDiff.gradient(multiz_pl_ap_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

SUITE["multiz_pipeline"]["gradient_forwarddiff_multiz_jac_ap"] = @benchmarkable begin
    ForwardDiff.gradient(multiz_jac_ap_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Zygote gradients
SUITE["multiz_pipeline"]["gradient_zygote_multiz_pl_ap"] = @benchmarkable begin
    Zygote.gradient(multiz_pl_ap_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

SUITE["multiz_pipeline"]["gradient_zygote_multiz_jac_ap"] = @benchmarkable begin
    Zygote.gradient(multiz_jac_ap_bench, cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)
