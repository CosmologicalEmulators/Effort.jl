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
using JSON

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

        jac_cosmology = DifferentiationInterface.jacobian(compute_P0_cosmology, AutoForwardDiff(), cosmology_params)
        @test all(isfinite.(jac_cosmology))
        @test size(jac_cosmology, 2) == length(cosmology_params)
        @test size(jac_cosmology, 1) > 0
    end

    @testset "ForwardDiff Jacobian - Bias Parameters" begin
        function compute_P0_bias(bias)
            return Effort.get_Pℓ(cosmology_params, D_growth, bias, monopole_emu)
        end

        jac_bias = DifferentiationInterface.jacobian(compute_P0_bias, AutoForwardDiff(), bias_params)
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

            grad_cosmology = DifferentiationInterface.gradient(loss_cosmology, AutoZygote(), cosmology_params)
            @test all(isfinite.(grad_cosmology))
            @test length(grad_cosmology) == length(cosmology_params)
        end

        @testset "Gradient w.r.t. bias parameters" begin
            function loss_bias(bias)
                P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias, monopole_emu)
                return sum(P0)
            end

            grad_bias = DifferentiationInterface.gradient(loss_bias, AutoZygote(), bias_params)
            @test all(isfinite.(grad_bias))
            @test length(grad_bias) == length(bias_params)
        end
    end

    # NOTE: Mooncake.jl does NOT work with Effort.jl's full emulator pipeline.
    # The trained emulators contain JSON.Object types in their Description metadata,
    # which causes StackOverflowError in Mooncake's tangent_type computation.
    # This is a fundamental limitation - even though the underlying LuxEmulator supports Mooncake,
    # the wrapped PℓEmulator structure with JSON metadata does not.

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
            grad0_fd = DifferentiationInterface.gradient(
                cosmology_params -> test_function(cosmology_params, monopole_emu),
                AutoForwardDiff(),
                cosmology_params
            )
            grad0_zy = DifferentiationInterface.gradient(
                cosmology_params -> test_function(cosmology_params, monopole_emu),
                AutoZygote(),
                cosmology_params
            )

            @test grad0_fd ≈ grad0_zy rtol=1e-5
        end

        @testset "Quadrupole consistency" begin
            grad2_fd = DifferentiationInterface.gradient(
                cosmology_params -> test_function(cosmology_params, quadrupole_emu),
                AutoForwardDiff(),
                cosmology_params
            )
            grad2_zy = DifferentiationInterface.gradient(
                cosmology_params -> test_function(cosmology_params, quadrupole_emu),
                AutoZygote(),
                cosmology_params
            )

            @test grad2_fd ≈ grad2_zy rtol=1e-5
        end

        @testset "Hexadecapole consistency" begin
            grad4_fd = DifferentiationInterface.gradient(
                cosmology_params -> test_function(cosmology_params, hexadecapole_emu),
                AutoForwardDiff(),
                cosmology_params
            )
            grad4_zy = DifferentiationInterface.gradient(
                cosmology_params -> test_function(cosmology_params, hexadecapole_emu),
                AutoZygote(),
                cosmology_params
            )

            @test grad4_fd ≈ grad4_zy rtol=1e-5
        end
    end

    # Mooncake tests removed: Mooncake.jl encounters StackOverflowError due to JSON.Object
    # types in emulator metadata. While the underlying LuxEmulator supports Mooncake,
    # Effort.jl's PℓEmulator wrapper contains Dict{String, JSON.Object{String, Any}} in the
    # Description field, which causes infinite recursion in Mooncake's tangent_type computation.
    # See benchmark/benchmarks.jl for Mooncake benchmarks on components that don't use emulator metadata.

    @testset "Multiple Multipoles Differentiation" begin
        function combined_multipole(cosmo_params)
            P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
            P2 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, quadrupole_emu)
            P4 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, hexadecapole_emu)
            return sum(P0 .^ 2 .+ P2 .^ 2 .+ P4 .^ 2)
        end

        @testset "ForwardDiff through all multipoles" begin
            grad_combined = DifferentiationInterface.gradient(combined_multipole, AutoForwardDiff(), cosmology_params)
            @test all(isfinite.(grad_combined))
            @test length(grad_combined) == length(cosmology_params)
        end

        @testset "Zygote through all multipoles" begin
            grad_combined_zy = DifferentiationInterface.gradient(combined_multipole, AutoZygote(), cosmology_params)
            @test all(isfinite.(grad_combined_zy))
            @test length(grad_combined_zy) == length(cosmology_params)
        end

        @testset "ForwardDiff vs Zygote for combined multipoles" begin
            grad_combined_fd = DifferentiationInterface.gradient(combined_multipole, AutoForwardDiff(), cosmology_params)
            grad_combined_zy = DifferentiationInterface.gradient(combined_multipole, AutoZygote(), cosmology_params)

            @test grad_combined_fd ≈ grad_combined_zy rtol=1e-5
        end
    end

    @testset "Bias Combination Jacobian" begin
        @testset "Monopole bias Jacobian" begin
            JFDb0 = DifferentiationInterface.jacobian(
                bias_params -> monopole_emu.BiasCombination(bias_params),
                AutoForwardDiff(),
                bias_params
            )
            Jb0 = monopole_emu.JacobianBiasCombination(bias_params)

            @test JFDb0 ≈ Jb0 rtol=1e-5
        end

        @testset "Quadrupole bias Jacobian" begin
            JFDb2 = DifferentiationInterface.jacobian(
                bias_params -> quadrupole_emu.BiasCombination(bias_params),
                AutoForwardDiff(),
                bias_params
            )
            Jb2 = quadrupole_emu.JacobianBiasCombination(bias_params)

            @test JFDb2 ≈ Jb2 rtol=1e-5
        end

        @testset "Hexadecapole bias Jacobian" begin
            JFDb4 = DifferentiationInterface.jacobian(
                bias_params -> hexadecapole_emu.BiasCombination(bias_params),
                AutoForwardDiff(),
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

    @testset "Loaded Emulator Metadata Verification and Mooncake Tests" begin
        # This testset verifies that:
        # 1. Emulators loaded from JSON files have properly converted metadata
        # 2. Mooncake.jl can differentiate through the loaded emulators
        # This ensures the JSON.Object → Dict{String, Any} conversion works correctly

        @testset "Metadata Conversion Verification" begin
            # Verify that loaded emulators have converted Description fields
            # (not JSON.Object types which would cause StackOverflowError in Mooncake)

            monopole_desc = monopole_emu.P11.TrainedEmulator.Description

            @test monopole_desc isa AbstractDict
            @test haskey(monopole_desc, "emulator_description")

            # The critical test: verify NO JSON.Object types remain
            emu_desc = monopole_desc["emulator_description"]
            @test !(typeof(emu_desc) <: JSON.Object)
            @test typeof(emu_desc) == Dict{String, Any}
        end

        @testset "Mooncake Backend with Loaded Emulators" begin
            # Test that Mooncake can differentiate through emulators loaded from JSON files
            # This would fail with StackOverflowError if JSON.Object conversion wasn't working

            function loss_cosmology_mooncake(cosmo_params)
                P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
                return sum(P0 .^ 2)
            end

            @testset "Mooncake gradient computation" begin
                # This is the critical test - Mooncake should work without StackOverflowError
                grad_mooncake = DifferentiationInterface.gradient(
                    loss_cosmology_mooncake,
                    AutoMooncake(; config=Mooncake.Config()),
                    cosmology_params
                )

                @test all(isfinite.(grad_mooncake))
                @test length(grad_mooncake) == length(cosmology_params)
            end

            @testset "Mooncake vs ForwardDiff consistency" begin
                # Verify Mooncake produces same results as ForwardDiff
                grad_fd = DifferentiationInterface.gradient(
                    loss_cosmology_mooncake,
                    AutoForwardDiff(),
                    cosmology_params
                )

                grad_mooncake = DifferentiationInterface.gradient(
                    loss_cosmology_mooncake,
                    AutoMooncake(; config=Mooncake.Config()),
                    cosmology_params
                )

                @test grad_fd ≈ grad_mooncake rtol=1e-5
            end

            @testset "Mooncake vs Zygote consistency" begin
                # Verify Mooncake produces same results as Zygote
                grad_zy = DifferentiationInterface.gradient(
                    loss_cosmology_mooncake,
                    AutoZygote(),
                    cosmology_params
                )

                grad_mooncake = DifferentiationInterface.gradient(
                    loss_cosmology_mooncake,
                    AutoMooncake(; config=Mooncake.Config()),
                    cosmology_params
                )

                @test grad_zy ≈ grad_mooncake rtol=1e-5
            end

            @testset "Mooncake with all multipoles" begin
                # Test Mooncake with combined multipole computation
                function combined_multipole_mooncake(cosmo_params)
                    P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
                    P2 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, quadrupole_emu)
                    P4 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, hexadecapole_emu)
                    return sum(P0 .^ 2 .+ P2 .^ 2 .+ P4 .^ 2)
                end

                grad_combined_mk = DifferentiationInterface.gradient(
                    combined_multipole_mooncake,
                    AutoMooncake(; config=Mooncake.Config()),
                    cosmology_params
                )

                @test all(isfinite.(grad_combined_mk))
                @test length(grad_combined_mk) == length(cosmology_params)

                # Verify consistency with ForwardDiff
                grad_combined_fd = DifferentiationInterface.gradient(
                    combined_multipole_mooncake,
                    AutoForwardDiff(),
                    cosmology_params
                )

                @test grad_combined_fd ≈ grad_combined_mk rtol=1e-5
            end
        end
    end
end
