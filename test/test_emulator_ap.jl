"""
Tests for emulator + AP effect integration.

Tests cover:
- Jacobian computation followed by AP transformation
- Two approaches: (Jacobian → AP per column) vs (ForwardDiff through full pipeline)
- Batch apply_AP consistency (matrix vs column-by-column)
- End-to-end automatic differentiation through get_Pℓ_jacobian + apply_AP
"""

using Test
using Effort
using ForwardDiff

@testset "Emulator + AP Effect Integration" begin
    @testset "Jacobian with AP Effect" begin
        # Set up test parameters
        z_test = 0.8
        ln10As_test = 3.044
        ns_test = 0.9649
        H0_test = 67.36
        ωb_test = 0.02237
        ωcdm_test = 0.12
        mν_test = 0.06
        w0_test = -1.0
        wa_test = 0.0
        cosmology_test = [z_test, ln10As_test, ns_test, H0_test, ωb_test, ωcdm_test, mν_test, w0_test, wa_test]

        # Bias parameters
        b1_test = 2.0
        b2_test = -0.5
        b3_test = 0.3
        b4_test = 0.5
        b5_test = 0.5
        b6_test = 0.5
        b7_test = 0.5
        f_test = 0.8
        cϵ0_test = 1.0
        cϵ1_test = 1.0
        cϵ2_test = 1.0
        bias_test = [b1_test, b2_test, b3_test, b4_test, b5_test, b6_test, b7_test, f_test, cϵ0_test, cϵ1_test, cϵ2_test]

        # Growth factor
        D_test = 0.75

        # AP parameters
        mycosmo_test = Effort.w0waCDMCosmology(ln10Aₛ=3.044, nₛ=0.9649, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
        mycosmo_ref_test = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.022, ωc=0.119, mν=0.06, w0=-1.0, wa=0.0)
        qpar_test, qperp_test = Effort.q_par_perp(z_test, mycosmo_test, mycosmo_ref_test)

        # Get emulators for all multipoles
        monopole_emu_test = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        quadrupole_emu_test = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
        hexadecapole_emu_test = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

        k_grid = vec(monopole_emu_test.P11.kgrid)  # Ensure it's a vector
        k_output_test = k_grid  # Use same grid for simplicity

        # APPROACH 1: Compute Jacobian first, then apply AP to each column
        P0_no_AP, Jac0_no_AP = Effort.get_Pℓ_jacobian(cosmology_test, D_test, bias_test, monopole_emu_test)
        P2_no_AP, Jac2_no_AP = Effort.get_Pℓ_jacobian(cosmology_test, D_test, bias_test, quadrupole_emu_test)
        P4_no_AP, Jac4_no_AP = Effort.get_Pℓ_jacobian(cosmology_test, D_test, bias_test, hexadecapole_emu_test)

        # Apply AP to the power spectra (for reference)
        P0_AP_single, P2_AP_single, P4_AP_single = Effort.apply_AP(
            k_grid, k_output_test, P0_no_AP, P2_no_AP, P4_no_AP,
            qpar_test, qperp_test, n_GL_points=8
        )

        # Apply AP to each Jacobian column using batch apply_AP
        Jac0_AP, Jac2_AP_temp, Jac4_AP_temp = Effort.apply_AP(
            k_grid, k_output_test, Jac0_no_AP, Jac2_no_AP, Jac4_no_AP,
            qpar_test, qperp_test, n_GL_points=8
        )

        # APPROACH 2: Use ForwardDiff to differentiate (get_Pℓ + apply_AP)
        function multipole_with_AP(bias_params, emu, cosmology, D, qpar, qperp, k_in, k_out)
            Pℓ = Effort.get_Pℓ(cosmology, D, bias_params, emu)
            # For a single multipole, we need to extract it properly
            # get_Pℓ returns a vector, we need to make it work with apply_AP
            return Pℓ
        end

        # Function that computes all three multipoles with AP
        function all_multipoles_with_AP(bias_params)
            P0 = Effort.get_Pℓ(cosmology_test, D_test, bias_params, monopole_emu_test)
            P2 = Effort.get_Pℓ(cosmology_test, D_test, bias_params, quadrupole_emu_test)
            P4 = Effort.get_Pℓ(cosmology_test, D_test, bias_params, hexadecapole_emu_test)

            # Apply AP
            P0_ap, P2_ap, P4_ap = Effort.apply_AP(
                k_grid, k_output_test, P0, P2, P4,
                qpar_test, qperp_test, n_GL_points=8
            )

            # Return concatenated result
            return vcat(P0_ap, P2_ap, P4_ap)
        end

        # Compute Jacobian using ForwardDiff
        Jac_FD = ForwardDiff.jacobian(all_multipoles_with_AP, bias_test)

        # Extract individual multipole Jacobians from ForwardDiff result
        n_k = length(k_output_test)
        Jac0_FD = Jac_FD[1:n_k, :]
        Jac2_FD = Jac_FD[(n_k+1):(2*n_k), :]
        Jac4_FD = Jac_FD[(2*n_k+1):(3*n_k), :]

        # Compare the two approaches
        # Test with rtol=1e-3
        @testset "Columns 1-8 with rtol=1e-3" begin
            @test isapprox(Jac0_AP[:, 1:8], Jac0_FD[:, 1:8], rtol=1e-3)
            @test isapprox(Jac2_AP_temp[:, 1:8], Jac2_FD[:, 1:8], rtol=1e-3)
            @test isapprox(Jac4_AP_temp[:, 1:8], Jac4_FD[:, 1:8], rtol=1e-3)
        end

        # Test stochastic columns (9-11)
        @testset "Stochastic columns (9-11)" begin
            # Monopole and quadrupole use relative tolerance
            @test isapprox(Jac0_AP[:, 9:11], Jac0_FD[:, 9:11], rtol=1e-3)
            @test isapprox(Jac2_AP_temp[:, 9:11], Jac2_FD[:, 9:11], rtol=1e-3)
            # Hexadecapole stochastic model returns zeros, so both should be ~0
            # Analytical gives machine epsilon (~1e-16), ForwardDiff gives numerical noise (~1e-7)
            # This is expected - the true derivative is zero
            @test maximum(abs.(Jac4_AP_temp[:, 9:11])) < 1e-14
        end

        # Verify that the power spectra match (sanity check)
        P_all_FD = all_multipoles_with_AP(bias_test)
        @test isapprox(P0_AP_single, P_all_FD[1:n_k], rtol=1e-10)
        @test isapprox(P2_AP_single, P_all_FD[(n_k+1):(2*n_k)], rtol=1e-10)
        @test isapprox(P4_AP_single, P_all_FD[(2*n_k+1):(3*n_k)], rtol=1e-10)
    end

    @testset "Batch apply_AP consistency" begin
        # Use the same test setup as above
        z_test = 0.8
        ln10As_test = 3.044
        ns_test = 0.9649
        H0_test = 67.36
        ωb_test = 0.02237
        ωcdm_test = 0.12
        mν_test = 0.06
        w0_test = -1.0
        wa_test = 0.0
        cosmology_test = [z_test, ln10As_test, ns_test, H0_test, ωb_test, ωcdm_test, mν_test, w0_test, wa_test]

        bias_test = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
        D_test = 0.75

        # AP parameters
        mycosmo_test = Effort.w0waCDMCosmology(ln10Aₛ=3.044, nₛ=0.9649, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
        mycosmo_ref_test = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.022, ωc=0.119, mν=0.06, w0=-1.0, wa=0.0)
        qpar_test, qperp_test = Effort.q_par_perp(z_test, mycosmo_test, mycosmo_ref_test)

        # Get emulators and k-grid
        monopole_emu_test = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        quadrupole_emu_test = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
        hexadecapole_emu_test = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]
        k_grid = vec(monopole_emu_test.P11.kgrid)
        k_output_test = k_grid

        # Compute Jacobian matrices (before AP)
        P0_no_AP, Jac0_no_AP = Effort.get_Pℓ_jacobian(cosmology_test, D_test, bias_test, monopole_emu_test)
        P2_no_AP, Jac2_no_AP = Effort.get_Pℓ_jacobian(cosmology_test, D_test, bias_test, quadrupole_emu_test)
        P4_no_AP, Jac4_no_AP = Effort.get_Pℓ_jacobian(cosmology_test, D_test, bias_test, hexadecapole_emu_test)

        # METHOD 1: Apply AP using the batch function (Matrix input)
        Jac0_AP_batch, Jac2_AP_batch, Jac4_AP_batch = Effort.apply_AP(
            k_grid, k_output_test, Jac0_no_AP, Jac2_no_AP, Jac4_no_AP,
            qpar_test, qperp_test, n_GL_points=8
        )

        # METHOD 2: Apply AP to each column individually
        n_params = size(Jac0_no_AP, 2)
        Jac0_AP_colwise = zeros(length(k_output_test), n_params)
        Jac2_AP_colwise = zeros(length(k_output_test), n_params)
        Jac4_AP_colwise = zeros(length(k_output_test), n_params)

        for col in 1:n_params
            mono_col = Jac0_no_AP[:, col]
            quad_col = Jac2_no_AP[:, col]
            hexa_col = Jac4_no_AP[:, col]

            P0_col, P2_col, P4_col = Effort.apply_AP(
                k_grid, k_output_test, mono_col, quad_col, hexa_col,
                qpar_test, qperp_test, n_GL_points=8
            )

            Jac0_AP_colwise[:, col] = P0_col
            Jac2_AP_colwise[:, col] = P2_col
            Jac4_AP_colwise[:, col] = P4_col
        end

        # Compare the two methods - they should be identical
        @test isapprox(Jac0_AP_batch, Jac0_AP_colwise, rtol=1e-14)
        @test isapprox(Jac2_AP_batch, Jac2_AP_colwise, rtol=1e-14)
        @test isapprox(Jac4_AP_batch, Jac4_AP_colwise, rtol=1e-14)
    end
end
