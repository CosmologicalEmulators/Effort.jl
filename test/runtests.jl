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
- test_pipeline.jl: End-to-end pipeline tests (single redshift)
- test_multiz_pipeline.jl: Multi-redshift pipeline tests (5 z-bins)

To run all tests:
    julia --project=. -e 'using Pkg; Pkg.test()'

To reproduce the multi-redshift group in the same package test environment:
    julia --project=. -e 'using Pkg; Pkg.test(test_args=["multiz"])'
"""

const groups = (
    core = ("ap_isotropic", "legendre", "window_convolution", "cosmology", "interpolation", "ap_effect", "emulator", "emulator_ap"),
    pipeline = ("pipeline",),
    multiz = ("multiz_pipeline",),
    operators = ("bias_jacobians", "velocileptors", "chebyshev_operator", "ap_window_chebyshev", "type_stability"),
    reactant = ("ext_reactant_effort", "ext_reactant_emulator", "ext_reactant_velocileptors"),
    quality = ("aqua",),
)

# Compiler/AD state can outlive a testset. Run the complete suite sequentially
# in fresh processes, keeping Pkg.test's resolved extras and Julia flags.
if isempty(ARGS)
    for group in keys(groups)
        run(`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $(@__FILE__) $(string(group))`)
    end
    exit(0)
end
group = Symbol(only(ARGS))
haskey(groups, group) || error("Unknown test group: $group. Choose one of $(keys(groups)).")
println("BEGIN Effort test group: $group"); flush(stdout)

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

# DifferentiationInterface for unified gradient API
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake, AutoFiniteDifferences
using Mooncake

if group === :core
  @testset "Projection API" begin
    @test :apply_AP in names(Effort)
    @test :apply_AP_check ∉ names(Effort)
    @test Effort.apply_AP_check isa Function
    @test !isdefined(Effort, :_transformed_weights)
end
end

# Load shared test fixtures and utilities
include("test_fixtures.jl")

# Include all test files
# Each file contains @testset blocks that will run automatically
for name in groups[group]
    println("BEGIN test_$name.jl"); flush(stdout)
    include("test_$name.jl")
end
println("PASS Effort test group: $group (peak RSS $(round(Sys.maxrss()/2^30; digits=2)) GiB)")
