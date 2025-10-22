"""
Test fixtures and shared test data for Effort.jl test suite.

This file contains common test data, fixtures, and helper functions
used across multiple test files.
"""

using NPZ
using SimpleChains
using Static
using Effort

# =============================================================================
# Neural Network Setup (for emulator tests)
# =============================================================================

const TEST_MLP = SimpleChain(
    static(6),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(identity, 40)
)

const TEST_K_GRID = Array(LinRange(0, 200, 40))
const TEST_WEIGHTS = SimpleChains.init_params(TEST_MLP)
const TEST_INMINMAX = rand(6, 2)
const TEST_OUTMINMAX = rand(40, 2)

# Cosmology parameters for testing
const TEST_COSMO_PARAMS = (a=1.0, Ωcb0=0.3, mν=0.06, h=0.67, w0=-1.1, wa=0.2)

# Test emulator objects
const TEST_EMU = Effort.SimpleChainsEmulator(Architecture=TEST_MLP, Weights=TEST_WEIGHTS)
const TEST_POSTPROCESSING = (input, output, D, Pkemu) -> output
const TEST_COMPONENT_EMU = Effort.ComponentEmulator(
    TrainedEmulator=TEST_EMU,
    kgrid=TEST_K_GRID,
    InMinMax=TEST_INMINMAX,
    OutMinMax=TEST_OUTMINMAX,
    Postprocessing=TEST_POSTPROCESSING
)

# =============================================================================
# Interpolation Test Data
# =============================================================================

const N_INTERP = 64
const INTERP_X1 = vcat([0.0], sort(rand(N_INTERP - 2)), [1.0])
const INTERP_X2 = 2 .* vcat([0.0], sort(rand(N_INTERP - 2)), [1.0])
const INTERP_Y = rand(N_INTERP)

# =============================================================================
# Window Convolution Test Data
# =============================================================================

const WINDOW_W_4D = rand(2, 20, 3, 10)
const WINDOW_V_2D = rand(20, 10)

# =============================================================================
# AP Effect Test Data
# =============================================================================

const AP_N_POINTS = 100
const AP_X = Array(LinRange(0.0, 1.0, AP_N_POINTS))
const AP_MONOPOLE = sin.(AP_X)
const AP_QUADRUPOLE = 0.5 .* cos.(AP_X)
const AP_HEXADECAPOLE = 0.1 .* cos.(2 .* AP_X)
const AP_Q_PAR = 1.4
const AP_Q_PERP = 0.6

# =============================================================================
# Legendre Polynomial Test Data
# =============================================================================

const LEGENDRE_X = Array(LinRange(-1.0, 1.0, 100))

# =============================================================================
# Cosmology Test Data
# =============================================================================

const TEST_COSMO = Effort.w0waCDMCosmology(
    ln10Aₛ=3.0, nₛ=0.96, h=0.636,
    ωb=0.02237, ωc=0.1, mν=0.06,
    w0=-2.0, wa=1.0
)

const TEST_COSMO_REF = Effort.w0waCDMCosmology(
    ln10Aₛ=3.0, nₛ=0.96, h=0.6736,
    ωb=0.02237, ωc=0.12, mν=0.06,
    w0=-1.0, wa=0.0
)

# =============================================================================
# Real Data Test Fixtures (Downloaded from Zenodo)
# =============================================================================

"""
Download and load real test data from Zenodo.
Called once at test setup.
"""
function load_real_data()
    # Download test data
    run(`wget https://zenodo.org/api/records/15244205/files-archive`)
    run(`unzip files-archive`)

    # Load data
    k = npzread("k.npy")
    k_test = npzread("k_test.npy")
    Pℓ = npzread("no_AP.npy")
    Pℓ_AP = npzread("yes_AP.npy")

    # Cleanup
    rm("files-archive")
    rm("k.npy")
    rm("k_test.npy")
    rm("no_AP.npy")
    rm("yes_AP.npy")

    return (k=k, k_test=k_test, Pℓ=Pℓ, Pℓ_AP=Pℓ_AP)
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
Compare DataInterpolations quadratic spline with Effort's implementation.
"""
function reference_quadratic_spline(y, x, xn)
    using DataInterpolations
    spline = QuadraticSpline(y, x; extrapolation=ExtrapolationType.Extension)
    return spline.(xn)
end

"""
Check if gradient is approximately zero (for tests where zero gradient is expected).
"""
function is_approximately_zero(x; atol=1e-14)
    return maximum(abs.(x)) < atol
end

# =============================================================================
# Emulator Test Parameters
# =============================================================================

const EMULATOR_TEST_Z = 1.2
const EMULATOR_TEST_LN10AS = 3.0
const EMULATOR_TEST_NS = 0.96
const EMULATOR_TEST_H0 = 67.36
const EMULATOR_TEST_ΩB = 0.022
const EMULATOR_TEST_ΩCDM = 0.12
const EMULATOR_TEST_MΝ = 0.06
const EMULATOR_TEST_W0 = -1.0
const EMULATOR_TEST_WA = 0.0

const EMULATOR_TEST_COSMOLOGY = [
    EMULATOR_TEST_Z, EMULATOR_TEST_LN10AS, EMULATOR_TEST_NS,
    EMULATOR_TEST_H0, EMULATOR_TEST_ΩB, EMULATOR_TEST_ΩCDM,
    EMULATOR_TEST_MΝ, EMULATOR_TEST_W0, EMULATOR_TEST_WA
]

const EMULATOR_TEST_BIAS = [1.5, 0.5, 0.2, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0]
const EMULATOR_TEST_D_GROWTH = 0.8

# =============================================================================
# Pipeline Test Parameters
# =============================================================================

const PIPELINE_TEST_Z = 0.8
const PIPELINE_TEST_COSMO_PARAMS = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
const PIPELINE_TEST_BIAS_PARAMS = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]

const PIPELINE_COSMO_REF = Effort.w0waCDMCosmology(
    ln10Aₛ=3.0, nₛ=0.96, h=0.67,
    ωb=0.022, ωc=0.119, mν=0.06,
    w0=-1.0, wa=0.0
)

# =============================================================================
# Jacobian Test Parameters
# =============================================================================

const JAC_TEST_Z = 0.8
const JAC_TEST_COSMOLOGY = [0.8, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
const JAC_TEST_BIAS = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
const JAC_TEST_D = 0.75

const JAC_TEST_COSMO_MCMC = Effort.w0waCDMCosmology(
    ln10Aₛ=3.044, nₛ=0.9649, h=0.6736,
    ωb=0.02237, ωc=0.12, mν=0.06,
    w0=-1.0, wa=0.0
)

const JAC_TEST_COSMO_REF = Effort.w0waCDMCosmology(
    ln10Aₛ=3.0, nₛ=0.96, h=0.67,
    ωb=0.022, ωc=0.119, mν=0.06,
    w0=-1.0, wa=0.0
)
