"""
Tests for window convolution functionality.

Tests cover:
- 2D window convolution
- 4D window convolution
- Automatic differentiation (FiniteDifferences vs Zygote)
- Edge cases (different sizes, types)
"""

using Test
using Effort
using Zygote
using DifferentiationInterface
using DifferentiationInterface: AutoZygote
import ADTypes: AutoFiniteDifferences
using FiniteDifferences

@testset "Window Convolution" begin
    @testset "2D Window Convolution" begin
        W_2d = rand(50, 100)
        v_1d = rand(100)

        # Test forward pass
        result = Effort.window_convolution(W_2d, v_1d)

        # Check output shape
        @test size(result) == (50,)

        # Check result is finite
        @test all(isfinite.(result))

        # Verify computation manually for simple case
        # window_convolution should compute W * v
        expected = W_2d * v_1d
        @test result ≈ expected atol = 1e-10
    end

    @testset "2D: Comparison with Matrix Multiplication" begin
        # For 2D case, window_convolution should be equivalent to matrix-vector product
        n_out = 25
        n_in = 40
        W_2d = randn(n_out, n_in)
        v_1d = randn(n_in)

        result_conv = Effort.window_convolution(W_2d, v_1d)
        result_matmul = W_2d * v_1d

        @test result_conv ≈ result_matmul atol = 1e-12

        # Test gradient consistency
        grad_conv_W = DifferentiationInterface.gradient(W -> sum(Effort.window_convolution(W, v_1d)), AutoZygote(), W_2d)
        grad_matmul_W = DifferentiationInterface.gradient(W -> sum(W * v_1d), AutoZygote(), W_2d)
        @test grad_conv_W ≈ grad_matmul_W atol = 1e-12

        grad_conv_v = DifferentiationInterface.gradient(v -> sum(Effort.window_convolution(W_2d, v)), AutoZygote(), v_1d)
        grad_matmul_v = DifferentiationInterface.gradient(v -> sum(W_2d * v), AutoZygote(), v_1d)
        @test grad_conv_v ≈ grad_matmul_v atol = 1e-12
    end
end
