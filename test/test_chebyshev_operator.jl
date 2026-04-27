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

@testset "Chebyshev Operator Optimization" begin
    # 1. Setup problem dimensions
    m, N = 50, 200
    K = 40
    x_min, x_max = 0.0, 10.0
    x_grid = range(x_min, x_max, length=N)
    
    # Generic linear operator (e.g., a window function)
    M = randn(m, N)
    
    # A smooth test function evaluated on the grid
    f_test(x) = sin(x) + 0.5 * cos(2x)
    v_grid = f_test.(x_grid)
    
    # 2. Precomputation
    op = prepare_chebyshev_operator(M, x_grid, x_min, x_max, K)
    
    @testset "Accuracy" begin
        # Evaluate function at Chebyshev nodes
        v_nodes = f_test.(op.plan.nodes[1]) # nodes is a tuple for N-dimensions
        
        y_direct = M * v_grid
        y_cheb = apply_chebyshev_operator(op, v_nodes)
        
        # Check accuracy (should be high for smooth functions)
        @test isapprox(y_cheb, y_direct, atol=1e-8)
    end
    
    @testset "AD Compatibility (DifferentiationInterface)" begin
        v0 = f_test.(op.plan.nodes[1])
        
        # Test function for gradients
        test_f(v) = sum(apply_chebyshev_operator(op, v))
        
        # Baseline: Finite Differences
        grad_fd = DifferentiationInterface.gradient(test_f, AutoFiniteDifferences(; fdm=central_fdm(5, 1)), v0)
        
        # ForwardDiff
        grad_forward = DifferentiationInterface.gradient(test_f, AutoForwardDiff(), v0)
        @test isapprox(grad_forward, grad_fd, atol=1e-8)
        
        # Zygote
        grad_zygote = DifferentiationInterface.gradient(test_f, AutoZygote(), v0)
        @test isapprox(grad_zygote, grad_fd, atol=1e-8)
        
        # Mooncake
        grad_mooncake = DifferentiationInterface.gradient(test_f, AutoMooncake(; config=nothing), v0)
        @test isapprox(grad_mooncake, grad_fd, atol=1e-8)
    end
    
    @testset "Batch Support" begin
        n_batch = 5
        v_batch = randn(length(op.plan.nodes[1]), n_batch)
        
        y_batch = apply_chebyshev_operator(op, v_batch)
        @test size(y_batch) == (m, n_batch)
        
        # Individual checks
        for i in 1:n_batch
            y_ind = apply_chebyshev_operator(op, v_batch[:, i])
            @test isapprox(y_batch[:, i], y_ind)
        end
    end
end
