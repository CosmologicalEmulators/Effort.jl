"""
Tests for type stability of Effort.jl functions using JET.jl and @inferred.
"""

using Test
using JET
using Effort
using LegendrePolynomials
using LinearAlgebra

# Include test fixtures if not already included
if !@isdefined(EMULATOR_TEST_COSMOLOGY)
    include("test_fixtures.jl")
end

@testset "Type Stability and Static Analysis" begin

    @testset "JET.jl Static Analysis" begin
        # Test optimization/type-stability of core components
        @test_opt target_modules=(Effort,) Effort.w0waCDMCosmology()
        
        cosmo = Effort.w0waCDMCosmology()
        z = 0.8
        
        # Background
        @test_opt target_modules=(Effort,) Effort.D_f_z(z, cosmo)
        @test_opt target_modules=(Effort,) Effort.q_par_perp(z, cosmo, cosmo)
        
        # Emulation
        monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        cosmology_params = EMULATOR_TEST_COSMOLOGY
        bias_params = EMULATOR_TEST_BIAS
        D_growth = EMULATOR_TEST_D_GROWTH
        
        @test_opt target_modules=(Effort,) Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
        @test_opt target_modules=(Effort,) Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)
        
        # Projection/AP
        k_input = collect(range(0.001, 0.5, length=100))
        k_output = collect(range(0.001, 0.4, length=50))
        mono, quad, hexa = rand(100), rand(100), rand(100)
        q_par, q_perp = 1.05, 0.95
        @test_opt target_modules=(Effort,) Effort.apply_AP(k_input, k_output, mono, quad, hexa, q_par, q_perp)
    end

    @testset "Explicit @inferred Checks" begin
        @testset "Legendre and Utilities" begin
            @test @inferred(Effort._Legendre_0(0.5)) isa Float64
            @test @inferred(Effort._Legendre_2(0.5)) isa Float64
            @test @inferred(Effort._Legendre_4(0.5)) isa Float64
        end

        @testset "Background Cosmology" begin
            cosmo = Effort.w0waCDMCosmology()
            z = 0.8
            
            @test @inferred(Effort.E_z(z, cosmo)) isa Float64
            @test @inferred(Effort.d̃A_z(z, cosmo)) isa Float64
            
            D, f = @inferred Effort.D_f_z(z, cosmo)
            @test D isa Float64
            @test f isa Float64
            
            q_par, q_perp = @inferred Effort.q_par_perp(z, cosmo, cosmo)
            @test q_par isa Float64
            @test q_perp isa Float64
        end

        @testset "Multi-Redshift Pipeline" begin
            cosmo = Effort.w0waCDMCosmology()
            zs = [0.5, 0.8, 1.1]
            
            Ds, fs = @inferred Effort.D_f_z(zs, cosmo)
            @test Ds isa Vector{Float64}
            @test fs isa Vector{Float64}
            @test length(Ds) == 3
        end
        
        @testset "Emulator Core Functions" begin
            monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
            cosmology_params = EMULATOR_TEST_COSMOLOGY
            bias_params = EMULATOR_TEST_BIAS
            D_growth = EMULATOR_TEST_D_GROWTH
            
            P0 = @inferred Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
            @test P0 isa AbstractVector{Float64}
            
            P0_jac, J0 = @inferred Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)
            @test P0_jac isa AbstractVector{Float64}
            @test J0 isa AbstractMatrix{Float64}
        end

        @testset "Projection and AP Effect" begin
            k_input = collect(range(0.001, 0.5, length=100))
            k_output = collect(range(0.001, 0.4, length=50))
            mono, quad, hexa = rand(100), rand(100), rand(100)
            q_par, q_perp = 1.05, 0.95
            
            P0_obs, P2_obs, P4_obs = @inferred Effort.apply_AP(k_input, k_output, mono, quad, hexa, q_par, q_perp)
            @test P0_obs isa Vector{Float64}
            @test P2_obs isa Vector{Float64}
            @test P4_obs isa Vector{Float64}
        end
@testset "Unified Chebyshev Architecture" begin
    # Prepare dummy data
    n_in = 100
    k_in = collect(range(0.001, 0.5, length=n_in))
    M = rand(50, n_in)

    # Test ChebyshevOperator
    K = 40
    op = @inferred Effort.prepare_chebyshev_operator(M, k_in, 0.001, 0.4, K)
    @test op isa Effort.ChebyshevOperator

    # Test apply_chebyshev_operator
    # The input size must match the number of nodes in the plan
    nodes = op.plan.nodes[1]
    p_vals = rand(length(nodes))
    coeffs = @inferred Effort.apply_chebyshev_operator(op, p_vals)
    @test coeffs isa Vector{Float64}

    # Test APWindowChebyshevPlan
    n_out = 50
    # window_matrix must be n_out x n_in (size of k_in)
    window_matrix = rand(n_out, n_in)
    # Signature: W0, W2, W4, k_dense, k_min, k_max, K
    plan = @inferred Effort.prepare_ap_window_chebyshev(window_matrix, window_matrix, window_matrix, k_in, 0.001, 0.4, K)
    @test plan isa Effort.APWindowChebyshevPlan
    # Test apply_AP_and_window
    q_par, q_perp = 1.05, 0.95
    mono, quad, hexa = rand(n_in), rand(n_in), rand(n_in)

    p0w, p2w, p4w = @inferred Effort.apply_AP_and_window(plan, k_in, mono, quad, hexa, q_par, q_perp)
    @test p0w isa Vector{Float64}
    @test p2w isa Vector{Float64}
    @test p4w isa Vector{Float64}
end    end
end
