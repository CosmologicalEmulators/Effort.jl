"""
Tests for Legendre polynomial implementations.

Tests cover:
- Correctness vs LegendrePolynomials.jl reference
- Automatic differentiation (Zygote vs ForwardDiff)
- Edge cases (boundaries, special values)
"""

using Test
using Effort
using ForwardDiff
using Zygote
using LegendrePolynomials

@testset "Legendre Polynomials" begin
    @testset "Correctness vs Reference Implementation" begin
        # Test at various points including boundaries
        test_points = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for x in test_points
            # L₀(x) = 1
            @test Effort._Legendre_0(x) ≈ Pl(x, 0) atol=1e-14
            @test Effort._Legendre_0(x) ≈ 1.0 atol=1e-14

            # L₂(x) = (3x² - 1)/2
            @test Effort._Legendre_2(x) ≈ Pl(x, 2) atol=1e-14
            expected_L2 = (3 * x^2 - 1) / 2
            @test Effort._Legendre_2(x) ≈ expected_L2 atol=1e-14

            # L₄(x) = (35x⁴ - 30x² + 3)/8
            @test Effort._Legendre_4(x) ≈ Pl(x, 4) atol=1e-14
            expected_L4 = (35 * x^4 - 30 * x^2 + 3) / 8
            @test Effort._Legendre_4(x) ≈ expected_L4 atol=1e-14
        end
    end

    @testset "Vectorized Operations" begin
        x_vec = LEGENDRE_X

        # Test broadcasting works
        L0_vec = Effort._Legendre_0.(x_vec)
        L2_vec = Effort._Legendre_2.(x_vec)
        L4_vec = Effort._Legendre_4.(x_vec)

        @test length(L0_vec) == length(x_vec)
        @test length(L2_vec) == length(x_vec)
        @test length(L4_vec) == length(x_vec)

        # Spot check some values
        @test all(L0_vec .≈ 1.0)  # L₀ is always 1

        # Check boundaries
        @test L2_vec[1] ≈ Pl(-1.0, 2) atol=1e-14  # x = -1
        @test L2_vec[end] ≈ Pl(1.0, 2) atol=1e-14  # x = 1
        @test L4_vec[1] ≈ Pl(-1.0, 4) atol=1e-14
        @test L4_vec[end] ≈ Pl(1.0, 4) atol=1e-14
    end

    @testset "Automatic Differentiation: Zygote vs ForwardDiff" begin
        x_vec = LEGENDRE_X

        # Test L₀ gradients (should be zero since L₀(x) = 1)
        grad_zygote_L0 = DifferentiationInterface.gradient(x -> sum(x .* Effort._Legendre_0.(x)), AutoZygote(), x_vec)
        grad_forwarddiff_L0 = DifferentiationInterface.gradient(x -> sum(x .* Pl.(x, 0)), AutoForwardDiff(), x_vec)

        @test grad_zygote_L0 ≈ grad_forwarddiff_L0 rtol=1e-9
        # L₀'(x) = 0, but we have x * L₀(x) = x, so gradient is 1
        @test all(grad_zygote_L0 .≈ 1.0)

        # Test L₂ gradients
        grad_zygote_L2 = DifferentiationInterface.gradient(x -> sum(Effort._Legendre_2.(x)), AutoZygote(), x_vec)
        grad_forwarddiff_L2 = DifferentiationInterface.gradient(x -> sum(Pl.(x, 2)), AutoForwardDiff(), x_vec)

        @test grad_zygote_L2 ≈ grad_forwarddiff_L2 rtol=1e-9

        # Test L₄ gradients
        grad_zygote_L4 = DifferentiationInterface.gradient(x -> sum(Effort._Legendre_4.(x)), AutoZygote(), x_vec)
        grad_forwarddiff_L4 = DifferentiationInterface.gradient(x -> sum(Pl.(x, 4)), AutoForwardDiff(), x_vec)

        @test grad_zygote_L4 ≈ grad_forwarddiff_L4 rtol=1e-9
    end

    @testset "Scalar Differentiation" begin
        # Test at a few specific points
        for x in [-0.8, 0.0, 0.8]
            # L₂ derivative: d/dx[(3x² - 1)/2] = 3x
            grad_L2_zy = DifferentiationInterface.gradient(x -> Effort._Legendre_2(x), AutoZygote(), x)
            grad_L2_fd = ForwardDiff.derivative(x -> Pl(x, 2), x)
            @test grad_L2_zy ≈ grad_L2_fd rtol=1e-12
            @test grad_L2_zy ≈ 3 * x rtol=1e-12

            # L₄ derivative: d/dx[(35x⁴ - 30x² + 3)/8] = (140x³ - 60x)/8 = (35x³ - 15x)/2
            grad_L4_zy = DifferentiationInterface.gradient(x -> Effort._Legendre_4(x), AutoZygote(), x)
            grad_L4_fd = ForwardDiff.derivative(x -> Pl(x, 4), x)
            @test grad_L4_zy ≈ grad_L4_fd rtol=1e-12
            expected_dL4 = (35 * x^3 - 15 * x) / 2
            @test grad_L4_zy ≈ expected_dL4 rtol=1e-12
        end
    end

    @testset "Edge Cases and Special Values" begin
        # Test at boundaries x = ±1
        @test Effort._Legendre_0(1.0) == 1.0
        @test Effort._Legendre_0(-1.0) == 1.0

        # L₂(±1) = 1
        @test Effort._Legendre_2(1.0) ≈ 1.0 atol=1e-14
        @test Effort._Legendre_2(-1.0) ≈ 1.0 atol=1e-14

        # L₄(±1) = 1
        @test Effort._Legendre_4(1.0) ≈ 1.0 atol=1e-14
        @test Effort._Legendre_4(-1.0) ≈ 1.0 atol=1e-14

        # Test at zero
        # L₀(0) = 1
        @test Effort._Legendre_0(0.0) == 1.0
        # L₂(0) = -1/2
        @test Effort._Legendre_2(0.0) ≈ -0.5 atol=1e-14
        # L₄(0) = 3/8
        @test Effort._Legendre_4(0.0) ≈ 3/8 atol=1e-14
    end

    @testset "Type Stability" begin
        # Test with Float32
        x_f32 = Float32(0.5)
        @test Effort._Legendre_0(x_f32) isa Float32
        @test Effort._Legendre_2(x_f32) isa Float32
        @test Effort._Legendre_4(x_f32) isa Float32

        # Test with Float64
        x_f64 = Float64(0.5)
        @test Effort._Legendre_0(x_f64) isa Float64
        @test Effort._Legendre_2(x_f64) isa Float64
        @test Effort._Legendre_4(x_f64) isa Float64
    end

    @testset "Orthogonality Properties" begin
        # Legendre polynomials are orthogonal on [-1, 1] with weight 1
        # ∫₋₁¹ Lₙ(x) Lₘ(x) dx = 0 if n ≠ m, and 2/(2n+1) if n = m

        # We can't test the full integral here, but we can test
        # that the polynomials don't always have the same sign
        # (which would violate orthogonality)
        x_vec = range(-1, 1, length=100)

        L2_vals = Effort._Legendre_2.(x_vec)
        L4_vals = Effort._Legendre_4.(x_vec)

        # L₂ and L₄ should have different sign patterns
        # (this is a weak test but catches major errors)
        product = L2_vals .* L4_vals
        @test any(product .< 0)  # Should have some negative products
        @test any(product .> 0)  # Should have some positive products
    end
end
