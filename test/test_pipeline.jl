"""
Tests for complete end-to-end pipeline.

Tests cover:
- Full physical pipeline: Cosmology → ODE (D, f) → Emulator → AP effect
- Extended pipeline: Differentiation w.r.t. both cosmological and bias parameters
- Pipeline with Jacobians: ODE → get_Pℓ_jacobian → apply_AP → differentiation
- Automatic differentiation through entire pipeline (ForwardDiff, Zygote, FiniteDifferences)
- Physical sensitivities and gradient verification
"""

using Test
using Effort
using ForwardDiff
using Zygote
using FiniteDifferences

@testset "Final Comprehensive Test: Full Pipeline with ODE-based Growth Factors" begin
    # This test demonstrates complete end-to-end differentiability through the entire pipeline:
    # Cosmology params → D & f from ODE → Multipole prediction → AP correction → Scalar loss
    # We verify that FiniteDifferences, ForwardDiff, and Zygote all produce consistent gradients

    # Load trained emulators
    monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

    # Fixed redshift (NOT a parameter to differentiate)
    z_fixed = 0.8

    # Reference cosmology for AP effect
    cosmo_ref = Effort.w0waCDMCosmology(
        ln10Aₛ = 3.0, nₛ = 0.96, h = 0.67,
        ωb = 0.022, ωc = 0.119, mν = 0.06,
        w0 = -1.0, wa = 0.0
    )

    # Fixed bias parameters
    bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]

    # Get k-grid from emulator
    k_grid = vec(monopole_emu.P11.kgrid)

    @testset "Complete Physical Pipeline" begin
        """
        This function represents the complete physical pipeline:
        1. Construct cosmology from parameters
        2. Compute growth factor D and growth rate f from ODE solver
        3. Predict power spectrum multipoles using emulators
        4. Apply Alcock-Paczynski corrections
        5. Sum all multipoles to create a scalar output

        This is differentiable w.r.t. cosmological parameters (ln10As, ns, H0, ωb, ωcdm, mν, w0, wa)
        but NOT w.r.t. redshift z (which is fixed).
        """
        function complete_pipeline(cosmo_params_vector)
            # Unpack cosmological parameters (WITHOUT z)
            ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
            h = H0 / 100.0

            # Construct cosmology object
            cosmology = Effort.w0waCDMCosmology(
                ln10Aₛ = ln10As, nₛ = ns, h = h,
                ωb = ωb, ωc = ωcdm, mν = mν,
                w0 = w0, wa = wa
            )

            # Compute growth factor D and growth rate f from ODE solver
            D = Effort.D_z(z_fixed, cosmology)
            f = Effort.f_z(z_fixed, cosmology)

            # Update bias parameters with computed f (8th element is the growth rate)
            # Avoid mutation for Zygote compatibility and type issues with ForwardDiff
            bias_with_f = [bias_params[1], bias_params[2], bias_params[3],
                          bias_params[4], bias_params[5], bias_params[6],
                          bias_params[7], f, bias_params[9], bias_params[10], bias_params[11]]

            # Build emulator input (emulator expects: [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa])
            emulator_params = [z_fixed, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            # Predict multipoles using emulators
            P0 = Effort.get_Pℓ(emulator_params, D, bias_with_f, monopole_emu)
            P2 = Effort.get_Pℓ(emulator_params, D, bias_with_f, quadrupole_emu)
            P4 = Effort.get_Pℓ(emulator_params, D, bias_with_f, hexadecapole_emu)

            # Compute AP parameters
            q_par, q_perp = Effort.q_par_perp(z_fixed, cosmology, cosmo_ref)

            # Apply Alcock-Paczynski effect
            P0_AP, P2_AP, P4_AP = Effort.apply_AP(
                k_grid, k_grid, P0, P2, P4,
                q_par, q_perp, n_GL_points=8
            )

            # Create scalar output by summing all AP-corrected multipoles
            scalar_output = sum(P0_AP) + sum(P2_AP) + sum(P4_AP)

            return scalar_output
        end

        # Test cosmological parameters (WITHOUT z - z is fixed at z_fixed)
        # Order: [ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
        cosmo_params_test = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]

        # Test forward pass
        scalar_value = complete_pipeline(cosmo_params_test)
        @test isfinite(scalar_value)
        @test scalar_value != 0.0

        # Compute gradients with three different AD methods
        grad_forwarddiff = ForwardDiff.gradient(complete_pipeline, cosmo_params_test)
        @test all(isfinite, grad_forwarddiff)
        @test length(grad_forwarddiff) == 8

        grad_zygote = Zygote.gradient(complete_pipeline, cosmo_params_test)[1]
        @test all(isfinite, grad_zygote)
        @test length(grad_zygote) == 8

        grad_finitediff = FiniteDifferences.grad(
            central_fdm(5, 1),
            complete_pipeline,
            cosmo_params_test
        )[1]
        @test all(isfinite, grad_finitediff)
        @test length(grad_finitediff) == 8

        # Compare gradients across methods (rtol=1e-3)
        @test isapprox(grad_forwarddiff, grad_zygote, rtol=1e-3)
        @test isapprox(grad_forwarddiff, grad_finitediff, rtol=1e-3)
        @test isapprox(grad_zygote, grad_finitediff, rtol=1e-3)

        # Verify physical sensitivities
        @test abs(grad_forwarddiff[1]) > abs(grad_forwarddiff[2])  # ln10As > ns
        @test count(x -> abs(x) > 1e-6, grad_forwarddiff) >= 6
        @test all(x -> abs(x) < 1e10, grad_forwarddiff)
        @test any(x -> abs(x) > 1e-6, grad_forwarddiff)
    end

    @testset "Extended Pipeline: Cosmological + Bias Parameters" begin
        """
        This test extends the complete pipeline to differentiate w.r.t. BOTH:
        - 8 cosmological parameters: [ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
        - 10 bias parameters: [b1, b2, b3, b4, cct, cr1, cr2, (f computed), ce0, cemono, cequad]

        Note: The growth rate f (8th bias parameter) is computed from the ODE solver,
        so we only pass 10 bias parameters and insert f at the correct position.

        Total input: 18 parameters (8 cosmological + 10 bias)
        """
        function complete_pipeline_extended(all_params)
            # Split parameters: first 8 are cosmological, next 10 are bias (excluding f)
            ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = all_params[1:8]
            bias_no_f = all_params[9:18]  # 10 bias parameters without f

            h = H0 / 100.0

            # Construct cosmology object
            cosmology = Effort.w0waCDMCosmology(
                ln10Aₛ = ln10As, nₛ = ns, h = h,
                ωb = ωb, ωc = ωcdm, mν = mν,
                w0 = w0, wa = wa
            )

            # Compute growth factor D and growth rate f from ODE solver
            D = Effort.D_z(z_fixed, cosmology)
            f = Effort.f_z(z_fixed, cosmology)

            # Construct full bias parameters with f inserted at position 8
            bias_with_f = [bias_no_f[1], bias_no_f[2], bias_no_f[3],
                          bias_no_f[4], bias_no_f[5], bias_no_f[6],
                          bias_no_f[7], f, bias_no_f[8], bias_no_f[9], bias_no_f[10]]

            # Build emulator input
            emulator_params = [z_fixed, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            # Predict multipoles using emulators
            P0 = Effort.get_Pℓ(emulator_params, D, bias_with_f, monopole_emu)
            P2 = Effort.get_Pℓ(emulator_params, D, bias_with_f, quadrupole_emu)
            P4 = Effort.get_Pℓ(emulator_params, D, bias_with_f, hexadecapole_emu)

            # Compute AP parameters
            q_par, q_perp = Effort.q_par_perp(z_fixed, cosmology, cosmo_ref)

            # Apply Alcock-Paczynski effect
            P0_AP, P2_AP, P4_AP = Effort.apply_AP(
                k_grid, k_grid, P0, P2, P4,
                q_par, q_perp, n_GL_points=8
            )

            # Create scalar output by summing all AP-corrected multipoles
            scalar_output = sum(P0_AP) + sum(P2_AP) + sum(P4_AP)

            return scalar_output
        end

        # Test parameters: 8 cosmological + 10 bias (excluding f)
        # Cosmological: [ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
        # Bias: [b1, b2, b3, b4, cct, cr1, cr2, ce0, cemono, cequad]
        all_params_test = [
            3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0,  # Cosmology (8)
            2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0      # Bias without f (10)
        ]

        # Test forward pass
        scalar_value = complete_pipeline_extended(all_params_test)
        @test isfinite(scalar_value)
        @test scalar_value != 0.0

        # Compute gradients with three different AD methods
        grad_forwarddiff = ForwardDiff.gradient(complete_pipeline_extended, all_params_test)
        @test all(isfinite, grad_forwarddiff)
        @test length(grad_forwarddiff) == 18  # 8 cosmological + 10 bias

        grad_zygote = Zygote.gradient(complete_pipeline_extended, all_params_test)[1]
        @test all(isfinite, grad_zygote)
        @test length(grad_zygote) == 18

        grad_finitediff = FiniteDifferences.grad(
            central_fdm(5, 1),
            complete_pipeline_extended,
            all_params_test
        )[1]
        @test all(isfinite, grad_finitediff)
        @test length(grad_finitediff) == 18

        # Compare gradients across methods (rtol=1e-3)
        @test isapprox(grad_forwarddiff, grad_zygote, rtol=1e-3)
        @test isapprox(grad_forwarddiff, grad_finitediff, rtol=1e-3)
        @test isapprox(grad_zygote, grad_finitediff, rtol=1e-3)

        # Verify all parameters have non-zero gradients (except possibly some bias params)
        @test count(x -> abs(x) > 1e-6, grad_forwarddiff) >= 15
        @test all(x -> abs(x) < 1e10, grad_forwarddiff)

        # Verify cosmological parameter gradients
        cosmo_grads = grad_forwarddiff[1:8]
        @test abs(cosmo_grads[1]) > abs(cosmo_grads[2])  # ln10As > ns

        # Verify bias parameter gradients exist
        bias_grads = grad_forwarddiff[9:18]
        @test any(x -> abs(x) > 1e-6, bias_grads)
    end

    @testset "Full Pipeline with Jacobians: ODE → Emulator → AP" begin
        """
        This test demonstrates the complete pipeline using Jacobians:
        1. Compute D and f from ODE solver (cosmology-dependent)
        2. Use get_Pℓ_jacobian to compute power spectrum Jacobians w.r.t. bias parameters
        3. Apply AP corrections to the Jacobians
        4. Sum all Jacobian elements to create a scalar output
        5. Differentiate w.r.t. cosmological parameters (NOT z)

        The Jacobian represents ∂Pℓ/∂(bias parameters), and we test that we can
        differentiate this Jacobian w.r.t. cosmological parameters.
        """
        function complete_pipeline_with_jacobians(cosmo_params_vector)
            # Unpack cosmological parameters (WITHOUT z)
            ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
            h = H0 / 100.0

            # Construct cosmology object
            cosmology = Effort.w0waCDMCosmology(
                ln10Aₛ = ln10As, nₛ = ns, h = h,
                ωb = ωb, ωc = ωcdm, mν = mν,
                w0 = w0, wa = wa
            )

            # Compute growth factor D and growth rate f from ODE solver
            D = Effort.D_z(z_fixed, cosmology)
            f = Effort.f_z(z_fixed, cosmology)

            # Update bias parameters with computed f
            bias_with_f = [bias_params[1], bias_params[2], bias_params[3],
                          bias_params[4], bias_params[5], bias_params[6],
                          bias_params[7], f, bias_params[9], bias_params[10], bias_params[11]]

            # Build emulator input
            emulator_params = [z_fixed, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            # Compute Jacobians using get_Pℓ_jacobian
            # Returns (Pℓ, Jacobian) where Jacobian has shape (n_k, n_bias)
            _, Jac0 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, monopole_emu)
            _, Jac2 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, quadrupole_emu)
            _, Jac4 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, hexadecapole_emu)

            # Compute AP parameters
            q_par, q_perp = Effort.q_par_perp(z_fixed, cosmology, cosmo_ref)

            # Apply Alcock-Paczynski effect to Jacobians
            # apply_AP works on matrices, so it applies to each column of the Jacobian
            Jac0_AP, Jac2_AP, Jac4_AP = Effort.apply_AP(
                k_grid, k_grid, Jac0, Jac2, Jac4,
                q_par, q_perp, n_GL_points=8
            )

            # Create scalar output by summing all AP-corrected Jacobian elements
            scalar_output = sum(Jac0_AP) + sum(Jac2_AP) + sum(Jac4_AP)

            return scalar_output
        end

        # Test cosmological parameters (WITHOUT z)
        # Order: [ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
        cosmo_params_test = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]

        # Test forward pass
        scalar_value = complete_pipeline_with_jacobians(cosmo_params_test)
        @test isfinite(scalar_value)
        @test scalar_value != 0.0

        # Compute gradients with ForwardDiff and FiniteDifferences
        # Note: Zygote encounters mutation issues with get_Pℓ_jacobian internals
        # but ForwardDiff and FiniteDifferences work correctly
        grad_forwarddiff = ForwardDiff.gradient(complete_pipeline_with_jacobians, cosmo_params_test)
        @test all(isfinite, grad_forwarddiff)
        @test length(grad_forwarddiff) == 8

        grad_finitediff = FiniteDifferences.grad(
            central_fdm(5, 1),
            complete_pipeline_with_jacobians,
            cosmo_params_test
        )[1]
        @test all(isfinite, grad_finitediff)
        @test length(grad_finitediff) == 8

        # Compare gradients across methods (rtol=1e-3)
        @test isapprox(grad_forwarddiff, grad_finitediff, rtol=1e-3)

        # Verify physical sensitivities
        @test abs(grad_forwarddiff[1]) > abs(grad_forwarddiff[2])  # ln10As > ns
        @test count(x -> abs(x) > 1e-6, grad_forwarddiff) >= 6
        @test all(x -> abs(x) < 1e10, grad_forwarddiff)
        @test any(x -> abs(x) > 1e-6, grad_forwarddiff)
    end
end
