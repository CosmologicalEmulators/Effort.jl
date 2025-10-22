"""
Main test runner for Effort.jl

This file orchestrates all tests by including individual test files.
Each test file is self-contained and tests a specific component.

Test organization:
- test_fixtures.jl: Shared test data and utilities
- test_legendre.jl: Legendre polynomial tests
- test_window_convolution.jl: Window convolution tests
- test_cosmology.jl: Background cosmology tests
- test_interpolation.jl: Interpolation methods (Akima)
- test_ap_effect.jl: Alcock-Paczynski transformation tests
- test_emulator.jl: Emulator and Jacobian tests
- test_emulator_ap.jl: Emulator + AP integration tests
- test_pipeline.jl: End-to-end pipeline tests

To run all tests:
    julia --project=. -e 'using Pkg; Pkg.test()'

To run specific test files during development:
    julia --project=. test/test_interpolation.jl
"""

# Package imports
using Test
using NPZ
using SimpleChains
using Static
using Effort
using ForwardDiff
using Zygote
using LegendrePolynomials
using FiniteDifferences
using SciMLSensitivity

# Load shared test fixtures and utilities
include("test_fixtures.jl")

# Load real test data from Zenodo (used by test_ap_effect.jl)
# This downloads and extracts test data, then returns the loaded arrays
real_data = load_real_data()

# Include all test files
# Each file contains @testset blocks that will run automatically
include("test_legendre.jl")
include("test_window_convolution.jl")
include("test_cosmology.jl")
include("test_interpolation.jl")
include("test_ap_effect.jl")
include("test_emulator.jl")
include("test_emulator_ap.jl")
include("test_pipeline.jl")
