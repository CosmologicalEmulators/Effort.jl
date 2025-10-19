using BenchmarkTools
using Effort
using AbstractCosmologicalEmulators
using LinearAlgebra
using DataInterpolations
using LegendrePolynomials

# Load extension dependencies for Background cosmology benchmarks
using OrdinaryDiffEqTsit5
using Integrals
using FastGaussQuadrature
using Zygote

# Every benchmark file must define a BenchmarkGroup named SUITE.
const SUITE = BenchmarkGroup()

# --- Load pretrained emulators for benchmarking ---
# The emulators are loaded in Effort's __init__ function
# We'll work with the PyBirdmnuw0wacdm emulators

# Define test input sizes and parameters
const nk_test = 100  # number of k points
const n_eft_params = 17  # number of EFT parameters

# Create test data
const k_test = collect(range(0.01, stop=0.3, length=nk_test))
const eft_params_test = randn(n_eft_params)

# --- Benchmark Groups ---
SUITE["emulator"] = BenchmarkGroup(["multipoles", "neural_network"])
SUITE["projection"] = BenchmarkGroup(["legendre", "AP_effect"])
SUITE["integration"] = BenchmarkGroup(["window_convolution"])
SUITE["background"] = BenchmarkGroup(["cosmology"])

# --- Multipole Emulator Benchmarks ---
# Benchmark running the monopole emulator (ℓ=0)
SUITE["emulator"]["monopole"] = @benchmarkable begin
    emulator_0 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    Effort.get_Pℓ(emulator_0, k_vals, params)
end setup = (
    k_vals = copy($k_test);
    params = copy($eft_params_test)
)

# Benchmark running the quadrupole emulator (ℓ=2)
SUITE["emulator"]["quadrupole"] = @benchmarkable begin
    emulator_2 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    Effort.get_Pℓ(emulator_2, k_vals, params)
end setup = (
    k_vals = copy($k_test);
    params = copy($eft_params_test)
)

# Benchmark running the hexadecapole emulator (ℓ=4)
SUITE["emulator"]["hexadecapole"] = @benchmarkable begin
    emulator_4 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]
    Effort.get_Pℓ(emulator_4, k_vals, params)
end setup = (
    k_vals = copy($k_test);
    params = copy($eft_params_test)
)

# Benchmark all three multipoles together
SUITE["emulator"]["all_multipoles"] = @benchmarkable begin
    emulator_0 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    emulator_2 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    emulator_4 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

    P0 = Effort.get_Pℓ(emulator_0, k_vals, params)
    P2 = Effort.get_Pℓ(emulator_2, k_vals, params)
    P4 = Effort.get_Pℓ(emulator_4, k_vals, params)
    (P0, P2, P4)
end setup = (
    k_vals = copy($k_test);
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

# Benchmark gradient of multipole evaluation
SUITE["gradients"]["multipole_gradient"] = @benchmarkable begin
    Zygote.gradient(params) do p
        emulator_0 = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        result = Effort.get_Pℓ(emulator_0, k_vals, p)
        sum(result)  # Need a scalar output for gradient
    end
end setup = (
    k_vals = collect(range(0.01, stop=0.1, length=20));
    params = randn($n_eft_params)
)