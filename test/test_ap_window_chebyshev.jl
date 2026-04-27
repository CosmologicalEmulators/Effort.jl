using Test
using Effort
using LinearAlgebra
using AbstractCosmologicalEmulators
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake, AutoFiniteDifferences
using Mooncake
using Zygote
using ForwardDiff
using FiniteDifferences

@testset "AP + Window Unified Chebyshev Architecture" begin
    # 1. Setup problem
    N_dense = 200
    k_min, k_max = 0.01, 0.4
    k_dense = collect(range(k_min, k_max, length=N_dense))
    K = 40
    
    # Mock window matrices (diagonal for simplicity in tests, but can be dense)
    W0 = diagm(ones(N_dense))
    W2 = diagm(ones(N_dense))
    W4 = diagm(ones(N_dense))
    
    # Mock input multipoles
    k_in = collect(range(0.005, 0.5, length=100))
    mono_in = exp.(-k_in / 0.1)
    quad_in = k_in .* exp.(-k_in / 0.1)
    hexa_in = k_in.^2 .* exp.(-k_in / 0.1)
    
    q_par, q_perp = 1.05, 0.95
    
    # 2. Precomputation
    plan = prepare_ap_window_chebyshev(W0, W2, W4, k_dense, k_min, k_max, K)
    
    @testset "Accuracy" begin
        # Traditional Baseline: evaluate AP on dense grid, then multiply by W
        mono_AP_dense, quad_AP_dense, hexa_AP_dense = apply_AP(k_in, k_dense, mono_in, quad_in, hexa_in, q_par, q_perp)
        
        y0_direct = W0 * mono_AP_dense
        y2_direct = W2 * quad_AP_dense
        y4_direct = W4 * hexa_AP_dense
        
        # New Optimized Path
        y0_opt, y2_opt, y4_opt = apply_AP_and_window(plan, k_in, mono_in, quad_in, hexa_in, q_par, q_perp)
        
        # Check agreement
        @test isapprox(y0_opt, y0_direct, atol=1e-5)
        @test isapprox(y2_opt, y2_direct, atol=1e-5)
        @test isapprox(y4_opt, y4_direct, atol=1e-5)
    end
    
    @testset "AD Compatibility" begin
        test_f(params) = begin
            q_pa, q_pe = params
            y0, y2, y4 = apply_AP_and_window(plan, k_in, mono_in, quad_in, hexa_in, q_pa, q_pe)
            return sum(y0) + sum(y2) + sum(y4)
        end
        
        params0 = [q_par, q_perp]
        
        # Baseline: Finite Differences
        grad_fd = DifferentiationInterface.gradient(test_f, AutoFiniteDifferences(; fdm=central_fdm(5, 1)), params0)
        
        # ForwardDiff
        grad_forward = DifferentiationInterface.gradient(test_f, AutoForwardDiff(), params0)
        @test isapprox(grad_forward, grad_fd, rtol=1e-6)
        
        # Zygote
        grad_zygote = DifferentiationInterface.gradient(test_f, AutoZygote(), params0)
        @test isapprox(grad_zygote, grad_fd, rtol=1e-6)
        
        # Mooncake
        grad_mooncake = DifferentiationInterface.gradient(test_f, AutoMooncake(; config=nothing), params0)
        @test isapprox(grad_mooncake, grad_fd, rtol=1e-6)
    end

    @testset "Batch Support" begin
        n_batch = 3
        mono_batch = mono_in * ones(1, n_batch) .+ 0.1 * randn(length(mono_in), n_batch)
        quad_batch = quad_in * ones(1, n_batch) .+ 0.1 * randn(length(quad_in), n_batch)
        hexa_batch = hexa_in * ones(1, n_batch) .+ 0.1 * randn(length(hexa_in), n_batch)

        y0_batch, y2_batch, y4_batch = apply_AP_and_window(plan, k_in, mono_batch, quad_batch, hexa_batch, q_par, q_perp)
        
        @test size(y0_batch) == (N_dense, n_batch)
        @test size(y2_batch) == (N_dense, n_batch)
        @test size(y4_batch) == (N_dense, n_batch)

        # Parity with individual applications
        # Relaxed atol slightly (1e-14) to account for matrix multiply associativity
        for i in 1:n_batch
            y0_i, y2_i, y4_i = apply_AP_and_window(plan, k_in, mono_batch[:, i], quad_batch[:, i], hexa_batch[:, i], q_par, q_perp)
            @test isapprox(y0_batch[:, i], y0_i, atol=1e-14)
            @test isapprox(y2_batch[:, i], y2_i, atol=1e-14)
            @test isapprox(y4_batch[:, i], y4_i, atol=1e-14)
        end

        # AD Compatibility (Zygote)
        test_f_batch(m_b) = begin
            y0, y2, y4 = apply_AP_and_window(plan, k_in, m_b, quad_batch, hexa_batch, q_par, q_perp)
            return sum(y0.^2) + sum(y2.^2) + sum(y4.^2)
        end
        
        grad_zygote = Zygote.gradient(test_f_batch, mono_batch)[1]
        @test size(grad_zygote) == size(mono_batch)
        @test all(isfinite.(grad_zygote))
    end
end
