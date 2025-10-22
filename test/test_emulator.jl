"""
Tests for emulator functionality and Jacobian computation.

Tests cover:
- Emulator loading and availability
- Basic multipole computation (get_Pℓ)
- Jacobian computation (get_Pℓ_jacobian)
- Automatic differentiation (ForwardDiff vs Zygote)
- Bias combination Jacobians
"""

using Test
using Effort
using ForwardDiff
using Zygote

@testset "Emulator and Jacobian Computation" begin
    @testset "Emulator Loading" begin
        # Test that emulators are properly loaded
        @test haskey(Effort.trained_emulators, "PyBirdmnuw0wacdm")
        @test haskey(Effort.trained_emulators["PyBirdmnuw0wacdm"], "0")
        @test haskey(Effort.trained_emulators["PyBirdmnuw0wacdm"], "2")
        @test haskey(Effort.trained_emulators["PyBirdmnuw0wacdm"], "4")

        # Get emulator references
        monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
        hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

        # Check that emulators are not nothing
        @test !isnothing(monopole_emu)
        @test !isnothing(quadrupole_emu)
        @test !isnothing(hexadecapole_emu)
    end

    # Set up test parameters (use fixtures)
    cosmology_params = EMULATOR_TEST_COSMOLOGY
    bias_params = EMULATOR_TEST_BIAS
    D_growth = EMULATOR_TEST_D_GROWTH

    # Get emulators
    monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

    @testset "Basic Multipole Computation" begin
        @testset "Monopole (ℓ=0)" begin
            P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
            @test length(P0) > 0
            @test all(isfinite.(P0))
            @test eltype(P0) <: AbstractFloat
        end

        @testset "Quadrupole (ℓ=2)" begin
            P2 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, quadrupole_emu)
            @test length(P2) > 0
            @test all(isfinite.(P2))
            @test eltype(P2) <: AbstractFloat
        end

        @testset "Hexadecapole (ℓ=4)" begin
            P4 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, hexadecapole_emu)
            @test length(P4) > 0
            @test all(isfinite.(P4))
            @test eltype(P4) <: AbstractFloat
        end

        @testset "All multipoles have same length" begin
            P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
            P2 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, quadrupole_emu)
            P4 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, hexadecapole_emu)

            @test length(P0) == length(P2) == length(P4)
        end
    end

    @testset "ForwardDiff Jacobian - Cosmology Parameters" begin
        function compute_P0_cosmology(cosmo_params)
            return Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
        end

        jac_cosmology = ForwardDiff.jacobian(compute_P0_cosmology, cosmology_params)
        @test all(isfinite.(jac_cosmology))
        @test size(jac_cosmology, 2) == length(cosmology_params)
        @test size(jac_cosmology, 1) > 0
    end

    @testset "ForwardDiff Jacobian - Bias Parameters" begin
        function compute_P0_bias(bias)
            return Effort.get_Pℓ(cosmology_params, D_growth, bias, monopole_emu)
        end

        jac_bias = ForwardDiff.jacobian(compute_P0_bias, bias_params)
        @test all(isfinite.(jac_bias))
        @test size(jac_bias, 2) == length(bias_params)
        @test size(jac_bias, 1) > 0
    end

    @testset "Zygote Gradient" begin
        @testset "Gradient w.r.t. cosmology parameters" begin
            function loss_cosmology(cosmo_params)
                P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
                return sum(P0)
            end

            grad_cosmology = Zygote.gradient(loss_cosmology, cosmology_params)[1]
            @test all(isfinite.(grad_cosmology))
            @test length(grad_cosmology) == length(cosmology_params)
        end

        @testset "Gradient w.r.t. bias parameters" begin
            function loss_bias(bias)
                P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias, monopole_emu)
                return sum(P0)
            end

            grad_bias = Zygote.gradient(loss_bias, bias_params)[1]
            @test all(isfinite.(grad_bias))
            @test length(grad_bias) == length(bias_params)
        end
    end

    @testset "Built-in Jacobian Function" begin
        @testset "Monopole Jacobian" begin
            P0, jac_P0 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)

            @test length(P0) > 0
            @test all(isfinite.(P0))
            @test size(jac_P0, 1) == length(P0)
            @test all(isfinite.(jac_P0))
        end

        @testset "Quadrupole Jacobian" begin
            P2, jac_P2 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, quadrupole_emu)

            @test all(isfinite.(P2))
            @test all(isfinite.(jac_P2))
            @test size(jac_P2, 1) == length(P2)
        end

        @testset "Hexadecapole Jacobian" begin
            P4, jac_P4 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, hexadecapole_emu)

            @test all(isfinite.(P4))
            @test all(isfinite.(jac_P4))
            @test size(jac_P4, 1) == length(P4)
        end

        @testset "Jacobian shape consistency" begin
            _, jac_P0 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)
            _, jac_P2 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, quadrupole_emu)
            _, jac_P4 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, hexadecapole_emu)

            # All Jacobians should have same shape (same k-grid, same number of parameters)
            @test size(jac_P0) == size(jac_P2) == size(jac_P4)
        end
    end

    @testset "ForwardDiff vs Zygote Consistency" begin
        function test_function(cosmo_params, emu)
            Pℓ = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, emu)
            return sum(Pℓ .^ 2)  # Sum of squares for a scalar output
        end

        @testset "Monopole consistency" begin
            grad0_fd = ForwardDiff.gradient(
                cosmology_params -> test_function(cosmology_params, monopole_emu),
                cosmology_params
            )
            grad0_zy = Zygote.gradient(
                cosmology_params -> test_function(cosmology_params, monopole_emu),
                cosmology_params
            )[1]

            @test grad0_fd ≈ grad0_zy rtol=1e-5
        end

        @testset "Quadrupole consistency" begin
            grad2_fd = ForwardDiff.gradient(
                cosmology_params -> test_function(cosmology_params, quadrupole_emu),
                cosmology_params
            )
            grad2_zy = Zygote.gradient(
                cosmology_params -> test_function(cosmology_params, quadrupole_emu),
                cosmology_params
            )[1]

            @test grad2_fd ≈ grad2_zy rtol=1e-5
        end

        @testset "Hexadecapole consistency" begin
            grad4_fd = ForwardDiff.gradient(
                cosmology_params -> test_function(cosmology_params, hexadecapole_emu),
                cosmology_params
            )
            grad4_zy = Zygote.gradient(
                cosmology_params -> test_function(cosmology_params, hexadecapole_emu),
                cosmology_params
            )[1]

            @test grad4_fd ≈ grad4_zy rtol=1e-5
        end
    end

    @testset "Multiple Multipoles Differentiation" begin
        function combined_multipole(cosmo_params)
            P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
            P2 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, quadrupole_emu)
            P4 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, hexadecapole_emu)
            return sum(P0 .^ 2 .+ P2 .^ 2 .+ P4 .^ 2)
        end

        @testset "ForwardDiff through all multipoles" begin
            grad_combined = ForwardDiff.gradient(combined_multipole, cosmology_params)
            @test all(isfinite.(grad_combined))
            @test length(grad_combined) == length(cosmology_params)
        end

        @testset "Zygote through all multipoles" begin
            grad_combined_zy = Zygote.gradient(combined_multipole, cosmology_params)[1]
            @test all(isfinite.(grad_combined_zy))
            @test length(grad_combined_zy) == length(cosmology_params)
        end

        @testset "ForwardDiff vs Zygote for combined multipoles" begin
            grad_combined_fd = ForwardDiff.gradient(combined_multipole, cosmology_params)
            grad_combined_zy = Zygote.gradient(combined_multipole, cosmology_params)[1]

            @test grad_combined_fd ≈ grad_combined_zy rtol=1e-5
        end
    end

    @testset "Bias Combination Jacobian" begin
        @testset "Monopole bias Jacobian" begin
            JFDb0 = ForwardDiff.jacobian(
                bias_params -> monopole_emu.BiasCombination(bias_params),
                bias_params
            )
            Jb0 = monopole_emu.JacobianBiasCombination(bias_params)

            @test JFDb0 ≈ Jb0 rtol=1e-5
        end

        @testset "Quadrupole bias Jacobian" begin
            JFDb2 = ForwardDiff.jacobian(
                bias_params -> quadrupole_emu.BiasCombination(bias_params),
                bias_params
            )
            Jb2 = quadrupole_emu.JacobianBiasCombination(bias_params)

            @test JFDb2 ≈ Jb2 rtol=1e-5
        end

        @testset "Hexadecapole bias Jacobian" begin
            JFDb4 = ForwardDiff.jacobian(
                bias_params -> hexadecapole_emu.BiasCombination(bias_params),
                bias_params
            )
            Jb4 = hexadecapole_emu.JacobianBiasCombination(bias_params)

            @test JFDb4 ≈ Jb4 rtol=1e-5
        end

        @testset "All bias Jacobians have correct shape" begin
            Jb0 = monopole_emu.JacobianBiasCombination(bias_params)
            Jb2 = quadrupole_emu.JacobianBiasCombination(bias_params)
            Jb4 = hexadecapole_emu.JacobianBiasCombination(bias_params)

            # All should have same number of columns (bias parameters)
            @test size(Jb0, 2) == size(Jb2, 2) == size(Jb4, 2) == length(bias_params)
        end
    end
end
