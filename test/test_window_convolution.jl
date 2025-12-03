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
    @testset "4D Window Convolution" begin
        W = WINDOW_W_4D  # (2, 20, 3, 10)
        v = WINDOW_V_2D   # (20, 10)

        # Test forward pass
        result = Effort.window_convolution(W, v)

        # Check output shape
        @test size(result) == (2, 3)

        # Check result is finite
        @test all(isfinite.(result))

        # Check it's not all zeros (sanity check)
        @test !all(result .== 0.0)
    end

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
        @test result ≈ expected atol=1e-10
    end

    @testset "Automatic Differentiation: Gradient w.r.t. v" begin
        W = WINDOW_W_4D
        v = WINDOW_V_2D

        # Gradient w.r.t. v using Zygote
        grad_zygote = DifferentiationInterface.gradient(v -> sum(Effort.window_convolution(W, v)), AutoZygote(), v)

        # Gradient w.r.t. v using FiniteDifferences
        grad_fd = DifferentiationInterface.gradient(
            v -> sum(Effort.window_convolution(W, v)),
            AutoFiniteDifferences(central_fdm(5, 1)),
            v
        )

        # Compare
        @test grad_zygote ≈ grad_fd rtol=1e-6
        @test size(grad_zygote) == size(v)
        @test all(isfinite.(grad_zygote))
    end

    @testset "Automatic Differentiation: Gradient w.r.t. W" begin
        W = WINDOW_W_4D
        v = WINDOW_V_2D

        # Gradient w.r.t. W using Zygote
        grad_zygote = DifferentiationInterface.gradient(W -> sum(Effort.window_convolution(W, v)), AutoZygote(), W)

        # Gradient w.r.t. W using FiniteDifferences
        grad_fd = DifferentiationInterface.gradient(
            W -> sum(Effort.window_convolution(W, v)),
            AutoFiniteDifferences(central_fdm(5, 1)),
            W
        )

        # Compare
        @test grad_zygote ≈ grad_fd rtol=1e-6
        @test size(grad_zygote) == size(W)
        @test all(isfinite.(grad_zygote))
    end

    @testset "Edge Cases: Different Sizes" begin
        # Test with minimal sizes
        W_min = rand(1, 2, 1, 3)
        v_min = rand(2, 3)
        result_min = Effort.window_convolution(W_min, v_min)
        @test size(result_min) == (1, 1)
        @test isfinite(result_min[1])

        # Test with larger sizes
        W_large = rand(5, 30, 4, 15)
        v_large = rand(30, 15)
        result_large = Effort.window_convolution(W_large, v_large)
        @test size(result_large) == (5, 4)
        @test all(isfinite.(result_large))
    end

    @testset "Type Stability" begin
        # Test with Float32
        W_f32 = Float32.(WINDOW_W_4D)
        v_f32 = Float32.(WINDOW_V_2D)
        result_f32 = Effort.window_convolution(W_f32, v_f32)
        @test eltype(result_f32) == Float32

        # Test with Float64
        W_f64 = Float64.(WINDOW_W_4D)
        v_f64 = Float64.(WINDOW_V_2D)
        result_f64 = Effort.window_convolution(W_f64, v_f64)
        @test eltype(result_f64) == Float64
    end

    @testset "Mathematical Properties" begin
        W = WINDOW_W_4D
        v = WINDOW_V_2D

        # Test linearity: conv(W, α*v) = α*conv(W, v)
        α = 2.5
        result1 = Effort.window_convolution(W, α * v)
        result2 = α * Effort.window_convolution(W, v)
        @test result1 ≈ result2 atol=1e-12

        # Test linearity: conv(α*W, v) = α*conv(W, v)
        result3 = Effort.window_convolution(α * W, v)
        result4 = α * Effort.window_convolution(W, v)
        @test result3 ≈ result4 atol=1e-12

        # Test additivity: conv(W, v1 + v2) = conv(W, v1) + conv(W, v2)
        v2 = rand(size(v)...)
        result5 = Effort.window_convolution(W, v + v2)
        result6 = Effort.window_convolution(W, v) + Effort.window_convolution(W, v2)
        @test result5 ≈ result6 atol=1e-12
    end

    @testset "Zero Cases" begin
        W = WINDOW_W_4D
        v = WINDOW_V_2D

        # Zero v should give zero result
        v_zero = zero(v)
        result_zero_v = Effort.window_convolution(W, v_zero)
        @test all(result_zero_v .≈ 0.0)

        # Zero W should give zero result
        W_zero = zero(W)
        result_zero_W = Effort.window_convolution(W_zero, v)
        @test all(result_zero_W .≈ 0.0)
    end

    @testset "2D: Comparison with Matrix Multiplication" begin
        # For 2D case, window_convolution should be equivalent to matrix-vector product
        n_out = 25
        n_in = 40
        W_2d = randn(n_out, n_in)
        v_1d = randn(n_in)

        result_conv = Effort.window_convolution(W_2d, v_1d)
        result_matmul = W_2d * v_1d

        @test result_conv ≈ result_matmul atol=1e-12

        # Test gradient consistency
        grad_conv_W = DifferentiationInterface.gradient(W -> sum(Effort.window_convolution(W, v_1d)), AutoZygote(), W_2d)
        grad_matmul_W = DifferentiationInterface.gradient(W -> sum(W * v_1d), AutoZygote(), W_2d)
        @test grad_conv_W ≈ grad_matmul_W atol=1e-12

        grad_conv_v = DifferentiationInterface.gradient(v -> sum(Effort.window_convolution(W_2d, v)), AutoZygote(), v_1d)
        grad_matmul_v = DifferentiationInterface.gradient(v -> sum(W_2d * v), AutoZygote(), v_1d)
        @test grad_conv_v ≈ grad_matmul_v atol=1e-12
    end
end
