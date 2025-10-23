"""
Multi-Redshift Pipeline Tests

Tests differentiation through realistic multi-redshift pipelines:
- 5 redshift bins from z=0.9 to z=1.8
- For each redshift: ODE (D, f) → Emulator → AP
- Tests all 4 configurations: Pℓ+AP, Pℓ+AP+Bias, Jac+AP, Jac+AP+Bias
- Compares ForwardDiff vs Zygote accuracy and performance
- Determines optimal tolerance for gradient agreement
"""

using Test
using Effort
using ForwardDiff
using Zygote
using FiniteDifferences

@testset "Multi-Redshift Pipeline Tests" begin
    # Load emulators
    monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

    # Multi-redshift setup: 5 bins from z=0.9 to z=1.8
    z_bins = range(0.9, 1.8, length=5) |> collect

    # Reference cosmology for AP effect
    cosmo_ref = Effort.w0waCDMCosmology(
        ln10Aₛ = 3.0, nₛ = 0.96, h = 0.67,
        ωb = 0.022, ωc = 0.119, mν = 0.06,
        w0 = -1.0, wa = 0.0
    )

    # Fixed bias parameters
    bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
    k_grid = vec(monopole_emu.P11.kgrid)

    #==========================================================================#
    # Helper Functions for Multi-Redshift Pipelines
    #==========================================================================#

    function multiz_pl_ap(cosmo_params_vector)
        ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
        h = H0 / 100.0

        cosmology = Effort.w0waCDMCosmology(
            ln10Aₛ = ln10As, nₛ = ns, h = h,
            ωb = ωb, ωc = ωcdm, mν = mν,
            w0 = w0, wa = wa
        )

        total = 0.0
        for z in z_bins
            D = Effort.D_z(z, cosmology)
            f = Effort.f_z(z, cosmology)

            bias_with_f = [bias_params[1], bias_params[2], bias_params[3],
                          bias_params[4], bias_params[5], bias_params[6],
                          bias_params[7], f, bias_params[9], bias_params[10], bias_params[11]]

            emulator_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            P0 = Effort.get_Pℓ(emulator_params, D, bias_with_f, monopole_emu)
            P2 = Effort.get_Pℓ(emulator_params, D, bias_with_f, quadrupole_emu)
            P4 = Effort.get_Pℓ(emulator_params, D, bias_with_f, hexadecapole_emu)

            q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

            P0_AP, P2_AP, P4_AP = Effort.apply_AP(
                k_grid, k_grid, P0, P2, P4,
                q_par, q_perp, n_GL_points=8
            )

            total += sum(P0_AP) + sum(P2_AP) + sum(P4_AP)
        end

        return total
    end

    function multiz_pl_ap_bias(all_params)
        ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = all_params[1:8]
        bias_no_f = all_params[9:18]
        h = H0 / 100.0

        cosmology = Effort.w0waCDMCosmology(
            ln10Aₛ = ln10As, nₛ = ns, h = h,
            ωb = ωb, ωc = ωcdm, mν = mν,
            w0 = w0, wa = wa
        )

        total = 0.0
        for z in z_bins
            D = Effort.D_z(z, cosmology)
            f = Effort.f_z(z, cosmology)

            bias_with_f = [bias_no_f[1], bias_no_f[2], bias_no_f[3],
                          bias_no_f[4], bias_no_f[5], bias_no_f[6],
                          bias_no_f[7], f, bias_no_f[8], bias_no_f[9], bias_no_f[10]]

            emulator_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            P0 = Effort.get_Pℓ(emulator_params, D, bias_with_f, monopole_emu)
            P2 = Effort.get_Pℓ(emulator_params, D, bias_with_f, quadrupole_emu)
            P4 = Effort.get_Pℓ(emulator_params, D, bias_with_f, hexadecapole_emu)

            q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

            P0_AP, P2_AP, P4_AP = Effort.apply_AP(
                k_grid, k_grid, P0, P2, P4,
                q_par, q_perp, n_GL_points=8
            )

            total += sum(P0_AP) + sum(P2_AP) + sum(P4_AP)
        end

        return total
    end

    function multiz_jac_ap(cosmo_params_vector)
        ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = cosmo_params_vector
        h = H0 / 100.0

        cosmology = Effort.w0waCDMCosmology(
            ln10Aₛ = ln10As, nₛ = ns, h = h,
            ωb = ωb, ωc = ωcdm, mν = mν,
            w0 = w0, wa = wa
        )

        total = 0.0
        for z in z_bins
            D = Effort.D_z(z, cosmology)
            f = Effort.f_z(z, cosmology)

            bias_with_f = [bias_params[1], bias_params[2], bias_params[3],
                          bias_params[4], bias_params[5], bias_params[6],
                          bias_params[7], f, bias_params[9], bias_params[10], bias_params[11]]

            emulator_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            _, Jac0 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, monopole_emu)
            _, Jac2 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, quadrupole_emu)
            _, Jac4 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, hexadecapole_emu)

            q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

            Jac0_AP, Jac2_AP, Jac4_AP = Effort.apply_AP(
                k_grid, k_grid, Jac0, Jac2, Jac4,
                q_par, q_perp, n_GL_points=8
            )

            total += sum(Jac0_AP) + sum(Jac2_AP) + sum(Jac4_AP)
        end

        return total
    end

    function multiz_jac_ap_bias(all_params)
        ln10As, ns, H0, ωb, ωcdm, mν, w0, wa = all_params[1:8]
        bias_no_f = all_params[9:18]
        h = H0 / 100.0

        cosmology = Effort.w0waCDMCosmology(
            ln10Aₛ = ln10As, nₛ = ns, h = h,
            ωb = ωb, ωc = ωcdm, mν = mν,
            w0 = w0, wa = wa
        )

        total = 0.0
        for z in z_bins
            D = Effort.D_z(z, cosmology)
            f = Effort.f_z(z, cosmology)

            bias_with_f = [bias_no_f[1], bias_no_f[2], bias_no_f[3],
                          bias_no_f[4], bias_no_f[5], bias_no_f[6],
                          bias_no_f[7], f, bias_no_f[8], bias_no_f[9], bias_no_f[10]]

            emulator_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

            _, Jac0 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, monopole_emu)
            _, Jac2 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, quadrupole_emu)
            _, Jac4 = Effort.get_Pℓ_jacobian(emulator_params, D, bias_with_f, hexadecapole_emu)

            q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

            Jac0_AP, Jac2_AP, Jac4_AP = Effort.apply_AP(
                k_grid, k_grid, Jac0, Jac2, Jac4,
                q_par, q_perp, n_GL_points=8
            )

            total += sum(Jac0_AP) + sum(Jac2_AP) + sum(Jac4_AP)
        end

        return total
    end

    # Test parameters
    params_8 = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    params_18 = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0,
                 2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

    #==========================================================================#
    # Test 1: Multi-z Pℓ + AP (8 cosmological parameters)
    #==========================================================================#

    @testset "Multi-z Pℓ + AP (8 params)" begin
        # Test forward pass
        result = multiz_pl_ap(params_8)
        @test isfinite(result)
        @test result != 0.0

        # Compute gradients
        grad_fd = ForwardDiff.gradient(multiz_pl_ap, params_8)
        grad_zy = Zygote.gradient(multiz_pl_ap, params_8)[1]
        grad_findiff = FiniteDifferences.grad(
            central_fdm(5, 1), multiz_pl_ap, params_8
        )[1]

        # Test gradient properties
        @test all(isfinite, grad_fd)
        @test all(isfinite, grad_zy)
        @test all(isfinite, grad_findiff)
        @test length(grad_fd) == 8
        @test length(grad_zy) == 8

        # Test with progressively stricter tolerances
        @test isapprox(grad_fd, grad_zy, rtol=1e-4)
        @test isapprox(grad_fd, grad_findiff, rtol=1e-3)
    end

    #==========================================================================#
    # Test 2: Multi-z Pℓ + AP + Bias (18 parameters)
    #==========================================================================#

    @testset "Multi-z Pℓ + AP + Bias (18 params)" begin
        # Test forward pass
        result = multiz_pl_ap_bias(params_18)
        @test isfinite(result)
        @test result != 0.0

        # Compute gradients
        grad_fd = ForwardDiff.gradient(multiz_pl_ap_bias, params_18)
        grad_zy = Zygote.gradient(multiz_pl_ap_bias, params_18)[1]
        grad_findiff = FiniteDifferences.grad(
            central_fdm(5, 1), multiz_pl_ap_bias, params_18
        )[1]

        # Test gradient properties
        @test all(isfinite, grad_fd)
        @test all(isfinite, grad_zy)
        @test all(isfinite, grad_findiff)
        @test length(grad_fd) == 18
        @test length(grad_zy) == 18

        # Test with progressively stricter tolerances
        @test isapprox(grad_fd, grad_zy, rtol=1e-4)
        @test isapprox(grad_fd, grad_findiff, rtol=1e-3)
    end

    #==========================================================================#
    # Test 3: Multi-z Jacobian + AP (8 cosmological parameters)
    #==========================================================================#

    @testset "Multi-z Jacobian + AP (8 params)" begin
        # Test forward pass
        result = multiz_jac_ap(params_8)
        @test isfinite(result)
        @test result != 0.0

        # Compute gradients
        grad_fd = ForwardDiff.gradient(multiz_jac_ap, params_8)
        grad_zy = Zygote.gradient(multiz_jac_ap, params_8)[1]
        grad_findiff = FiniteDifferences.grad(
            central_fdm(5, 1), multiz_jac_ap, params_8
        )[1]

        # Test gradient properties
        @test all(isfinite, grad_fd)
        @test all(isfinite, grad_zy)
        @test all(isfinite, grad_findiff)
        @test length(grad_fd) == 8
        @test length(grad_zy) == 8

        # Test with progressively stricter tolerances
        @test isapprox(grad_fd, grad_zy, rtol=1e-4)
        @test isapprox(grad_fd, grad_findiff, rtol=1e-3)
    end

    #==========================================================================#
    # Test 4: Multi-z Jacobian + AP + Bias (18 parameters) - FULL PIPELINE
    #==========================================================================#

    @testset "Multi-z Jacobian + AP + Bias (18 params)" begin
        # Test forward pass
        result = multiz_jac_ap_bias(params_18)
        @test isfinite(result)
        @test result != 0.0

        # Compute gradients
        grad_fd = ForwardDiff.gradient(multiz_jac_ap_bias, params_18)
        grad_zy = Zygote.gradient(multiz_jac_ap_bias, params_18)[1]
        grad_findiff = FiniteDifferences.grad(
            central_fdm(5, 1), multiz_jac_ap_bias, params_18
        )[1]

        # Test gradient properties
        @test all(isfinite, grad_fd)
        @test all(isfinite, grad_zy)
        @test all(isfinite, grad_findiff)
        @test length(grad_fd) == 18
        @test length(grad_zy) == 18

        # Count non-zero gradients
        @test count(x -> abs(x) > 1e-6, grad_fd) >= 16

        # Test with progressively stricter tolerances
        @test isapprox(grad_fd, grad_zy, rtol=1e-4)
        @test isapprox(grad_fd, grad_findiff, rtol=1e-4)

        # Based on measured error, test tighter tolerance
        #if max_rel_err_zy < 1e-4
        #    @test isapprox(grad_fd, grad_zy, rtol=1e-4)
        #end
        #if max_rel_err_zy < 1e-5
        #    @test isapprox(grad_fd, grad_zy, rtol=1e-5)
        #end
    end
end
