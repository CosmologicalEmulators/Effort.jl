using BenchmarkTools

using Zygote

# DifferentiationInterface and AD backends for gradient benchmarks
using DifferentiationInterface
using DifferentiationInterface: prepare_gradient
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake
using Mooncake
using ForwardDiff
using Effort

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

# --- Window Convolution Benchmarks ---
SUITE["integration"]["window_2D"] = @benchmarkable begin
    Effort.window_convolution(W, v)
end setup = (
    W = randn(50, 1000);
    v = randn(1000)
)

# --- Gradient Benchmarks (using DifferentiationInterface) ---
SUITE["gradients"] = BenchmarkGroup(["AD", "DifferentiationInterface"])

# Loss function for component gradient
component_loss(params, D, model) = sum(Effort.get_component(params, D, model))

# Benchmark gradient of component evaluation - Zygote
SUITE["gradients"]["component_gradient_zygote"] = @benchmarkable begin
    gradient!(f, grad_zygote, prep_zygote, backend_zygote, params)
end setup = (
    backend_zygote = AutoZygote();
    D = 1.0;
    model = $emulator_0.P11;
    f = p -> component_loss(p, D, model);
    typical_x = randn($n_eft_params);
    prep_zygote = prepare_gradient(f, backend_zygote, typical_x);
    params = randn($n_eft_params);
    grad_zygote = similar(params)
)

# Benchmark gradient of component evaluation - ForwardDiff
SUITE["gradients"]["component_gradient_forwarddiff"] = @benchmarkable begin
    gradient!(f, grad_fd, prep_fd, backend_fd, params)
end setup = (
    backend_fd = AutoForwardDiff();
    D = 1.0;
    model = $emulator_0.P11;
    f = p -> component_loss(p, D, model);
    typical_x = randn($n_eft_params);
    prep_fd = prepare_gradient(f, backend_fd, typical_x);
    params = randn($n_eft_params);
    grad_fd = similar(params)
)

# Benchmark gradient of component evaluation - Mooncake
SUITE["gradients"]["component_gradient_mooncake"] = @benchmarkable begin
    gradient!(f, grad_mc, prep_mc, backend_mc, params)
end setup = (
    backend_mc = AutoMooncake();
    D = 1.0;
    model = $emulator_0.P11;
    f = p -> component_loss(p, D, model);
    typical_x = randn($n_eft_params);
    prep_mc = prepare_gradient(f, backend_mc, typical_x);
    params = randn($n_eft_params);
    grad_mc = similar(params)
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
SUITE["full_pipeline"]["gradient_forwarddiff"] = @benchmarkable begin
    gradient!(complete_pipeline_bench, grad_fd, prep_fd, backend_fd, cosmo_params)
end setup = (
    backend_fd = AutoForwardDiff();
    typical_x = copy($cosmo_params_pipeline);
    prep_fd = prepare_gradient(complete_pipeline_bench, backend_fd, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_fd = similar(cosmo_params)
)

# Benchmark: Zygote gradient through complete pipeline
# This works with power spectra (Zygote has issues with Jacobian internals)
SUITE["full_pipeline"]["gradient_zygote"] = @benchmarkable begin
    gradient!(complete_pipeline_bench, grad_zygote, prep_zygote, backend_zygote, cosmo_params)
end setup = (
    backend_zygote = AutoZygote();
    typical_x = copy($cosmo_params_pipeline);
    prep_zygote = prepare_gradient(complete_pipeline_bench, backend_zygote, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_zygote = similar(cosmo_params)
)

# Benchmark: Mooncake gradient through complete pipeline
SUITE["full_pipeline"]["gradient_mooncake"] = @benchmarkable begin
    gradient!(complete_pipeline_bench, grad_mc, prep_mc, backend_mc, cosmo_params)
end setup = (
    backend_mc = AutoMooncake();
    typical_x = copy($cosmo_params_pipeline);
    prep_mc = prepare_gradient(complete_pipeline_bench, backend_mc, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_mc = similar(cosmo_params)
)

# Benchmark: Forward pass with Jacobians (ODE → Jacobian → AP)
SUITE["full_pipeline"]["forward_pass_jacobians"] = @benchmarkable begin
    complete_pipeline_jacobians_bench(cosmo_params)
end setup = (
    cosmo_params = copy($cosmo_params_pipeline)
)

# Benchmark: ForwardDiff gradient through Jacobian pipeline
SUITE["full_pipeline"]["gradient_forwarddiff_jacobians"] = @benchmarkable begin
    gradient!(complete_pipeline_jacobians_bench, grad_fd_jac, prep_fd_jac, backend_fd_jac, cosmo_params)
end setup = (
    backend_fd_jac = AutoForwardDiff();
    typical_x = copy($cosmo_params_pipeline);
    prep_fd_jac = prepare_gradient(complete_pipeline_jacobians_bench, backend_fd_jac, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_fd_jac = similar(cosmo_params)
)

# Benchmark: Zygote gradient through Jacobian pipeline (now works!)
SUITE["full_pipeline"]["gradient_zygote_jacobians"] = @benchmarkable begin
    gradient!(complete_pipeline_jacobians_bench, grad_zygote_jac, prep_zygote_jac, backend_zygote_jac, cosmo_params)
end setup = (
    backend_zygote_jac = AutoZygote();
    typical_x = copy($cosmo_params_pipeline);
    prep_zygote_jac = prepare_gradient(complete_pipeline_jacobians_bench, backend_zygote_jac, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_zygote_jac = similar(cosmo_params)
)

# Benchmark: Mooncake gradient through Jacobian pipeline
SUITE["full_pipeline"]["gradient_mooncake_jacobians"] = @benchmarkable begin
    gradient!(complete_pipeline_jacobians_bench, grad_mc_jac, prep_mc_jac, backend_mc_jac, cosmo_params)
end setup = (
    backend_mc_jac = AutoMooncake();
    typical_x = copy($cosmo_params_pipeline);
    prep_mc_jac = prepare_gradient(complete_pipeline_jacobians_bench, backend_mc_jac, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_mc_jac = similar(cosmo_params)
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
    gradient!(multiz_pl_ap_bench, grad_fd_multiz_pl, prep_fd_multiz_pl, backend_fd_multiz_pl, cosmo_params)
end setup = (
    backend_fd_multiz_pl = AutoForwardDiff();
    typical_x = copy($cosmo_params_pipeline);
    prep_fd_multiz_pl = prepare_gradient(multiz_pl_ap_bench, backend_fd_multiz_pl, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_fd_multiz_pl = similar(cosmo_params)
)

SUITE["multiz_pipeline"]["gradient_forwarddiff_multiz_jac_ap"] = @benchmarkable begin
    gradient!(multiz_jac_ap_bench, grad_fd_multiz_jac, prep_fd_multiz_jac, backend_fd_multiz_jac, cosmo_params)
end setup = (
    backend_fd_multiz_jac = AutoForwardDiff();
    typical_x = copy($cosmo_params_pipeline);
    prep_fd_multiz_jac = prepare_gradient(multiz_jac_ap_bench, backend_fd_multiz_jac, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_fd_multiz_jac = similar(cosmo_params)
)

# Zygote gradients
SUITE["multiz_pipeline"]["gradient_zygote_multiz_pl_ap"] = @benchmarkable begin
    gradient!(multiz_pl_ap_bench, grad_zygote_multiz_pl, prep_zygote_multiz_pl, backend_zygote_multiz_pl, cosmo_params)
end setup = (
    backend_zygote_multiz_pl = AutoZygote();
    typical_x = copy($cosmo_params_pipeline);
    prep_zygote_multiz_pl = prepare_gradient(multiz_pl_ap_bench, backend_zygote_multiz_pl, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_zygote_multiz_pl = similar(cosmo_params)
)

SUITE["multiz_pipeline"]["gradient_zygote_multiz_jac_ap"] = @benchmarkable begin
    gradient!(multiz_jac_ap_bench, grad_zygote_multiz_jac, prep_zygote_multiz_jac, backend_zygote_multiz_jac, cosmo_params)
end setup = (
    backend_zygote_multiz_jac = AutoZygote();
    typical_x = copy($cosmo_params_pipeline);
    prep_zygote_multiz_jac = prepare_gradient(multiz_jac_ap_bench, backend_zygote_multiz_jac, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_zygote_multiz_jac = similar(cosmo_params)
)

# Mooncake gradients
SUITE["multiz_pipeline"]["gradient_mooncake_multiz_pl_ap"] = @benchmarkable begin
    gradient!(multiz_pl_ap_bench, grad_mc_multiz_pl, prep_mc_multiz_pl, backend_mc_multiz_pl, cosmo_params)
end setup = (
    backend_mc_multiz_pl = AutoMooncake();
    typical_x = copy($cosmo_params_pipeline);
    prep_mc_multiz_pl = prepare_gradient(multiz_pl_ap_bench, backend_mc_multiz_pl, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_mc_multiz_pl = similar(cosmo_params)
)

SUITE["multiz_pipeline"]["gradient_mooncake_multiz_jac_ap"] = @benchmarkable begin
    gradient!(multiz_jac_ap_bench, grad_mc_multiz_jac, prep_mc_multiz_jac, backend_mc_multiz_jac, cosmo_params)
end setup = (
    backend_mc_multiz_jac = AutoMooncake();
    typical_x = copy($cosmo_params_pipeline);
    prep_mc_multiz_jac = prepare_gradient(multiz_jac_ap_bench, backend_mc_multiz_jac, typical_x);
    cosmo_params = copy($cosmo_params_pipeline);
    grad_mc_multiz_jac = similar(cosmo_params)
)
