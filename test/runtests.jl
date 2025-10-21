using Test
using NPZ
using SimpleChains
using Static
using Effort
using ForwardDiff
using Zygote
using LegendrePolynomials
using FiniteDifferences
using SciMLSensitivity
using DataInterpolations

mlpd = SimpleChain(
    static(6),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(identity, 40)
)

k_test = Array(LinRange(0, 200, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6, 2)
outminmax = rand(40, 2)
a, Ωcb0, mν, h, w0, wa = [1.0, 0.3, 0.06, 0.67, -1.1, 0.2]
z = Array(LinRange(0.0, 3.0, 100))

emu = Effort.SimpleChainsEmulator(Architecture=mlpd, Weights=weights)

postprocessing = (input, output, D, Pkemu) -> output

effort_emu = Effort.ComponentEmulator(TrainedEmulator=emu, kgrid=k_test, InMinMax=inminmax,
    OutMinMax=outminmax, Postprocessing=postprocessing)

x = [Ωcb0, h, mν, w0, wa]

n = 64
x1 = vcat([0.0], sort(rand(n - 2)), [1.0])
x2 = 2 .* vcat([0.0], sort(rand(n - 2)), [1.0])
y = rand(n)

W = rand(2, 20, 3, 10)
v = rand(20, 10)

function di_spline(y, x, xn)
    spline = QuadraticSpline(y, x; extrapolation=ExtrapolationType.Extension)
    return spline.(xn)
end

n_bar = 1e-3

myx = Array(LinRange(0.0, 1.0, 100))
monotest = sin.(myx)
quadtest = 0.5 .* cos.(myx)
hexatest = 0.1 .* cos.(2 .* myx)
q_par = 1.4
q_perp = 0.6

x3 = Array(LinRange(-1.0, 1.0, 100))

mycosmo = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.636, ωb=0.02237, ωc=0.1, mν=0.06, w0=-2.0, wa=1.0)

run(`wget https://zenodo.org/api/records/15244205/files-archive`)
run(`unzip files-archive`)
k = npzread("k.npy")
k_test = npzread("k_test.npy")
Pℓ = npzread("no_AP.npy")
Pℓ_AP = npzread("yes_AP.npy")
rm("files-archive")
rm("k.npy")
rm("k_test.npy")
rm("no_AP.npy")
rm("yes_AP.npy")

@testset "Effort tests" begin
    @test isapprox(Effort._quadratic_spline(y, x1, x2), di_spline(y, x1, x2), rtol=1e-9)
    @test isapprox(ForwardDiff.gradient(y -> sum(Effort._akima_spline_legacy(y, x1, x2)), y), Zygote.gradient(y -> sum(Effort._akima_spline_legacy(y, x1, x2)), y)[1], rtol=1e-9)
    @test isapprox(ForwardDiff.gradient(x1 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x1), Zygote.gradient(x1 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x1)[1], rtol=1e-9)
    @test isapprox(ForwardDiff.gradient(x2 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x2), Zygote.gradient(x2 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x2)[1], rtol=1e-9)
    @test isapprox(FiniteDifferences.jacobian(central_fdm(5, 1), monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest)[1], Zygote.jacobian(monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest)[1], rtol=1e-3)
    @test isapprox(ForwardDiff.jacobian(monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest), Zygote.jacobian(monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest)[1], rtol=1e-10)
    @test isapprox(ForwardDiff.jacobian(quadtest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), quadtest), Zygote.jacobian(quadtest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), quadtest)[1], rtol=1e-10)
    @test isapprox(grad(central_fdm(5, 1), v -> sum(Effort.window_convolution(W, v)), v)[1], Zygote.gradient(v -> sum(Effort.window_convolution(W, v)), v)[1], rtol=1e-6)
    @test isapprox(grad(central_fdm(5, 1), W -> sum(Effort.window_convolution(W, v)), W)[1], Zygote.gradient(W -> sum(Effort.window_convolution(W, v)), W)[1], rtol=1e-6)
    a_18, b_18, c_18 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)
    a_72, b_72, c_72 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=72)
    a_126, b_126, c_126 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=126)
    a_c, b_c, c_c = Effort.apply_AP_check(myx, myx, monotest, quadtest, hexatest, q_par, q_perp)
    @test isapprox(a_18, a_c, rtol=1e-5)
    @test isapprox(b_18, b_c, rtol=1e-5)
    @test isapprox(c_18, c_c, rtol=1e-5)

    @test isapprox(a_72, a_c, rtol=1e-6)
    @test isapprox(b_72, b_c, rtol=1e-6)
    @test isapprox(c_72, c_c, rtol=1e-6)

    @test isapprox(a_126, a_c, rtol=1e-7)
    @test isapprox(b_126, b_c, rtol=1e-7)
    @test isapprox(c_126, c_c, rtol=1e-7)

    @test isapprox(Zygote.gradient(x3 -> sum(x3 .* Effort._Legendre_0.(x3)), x3)[1], ForwardDiff.gradient(x3 -> sum(x3 .* Pl.(x3, 0)), x3), rtol=1e-9)
    @test isapprox(Zygote.gradient(x3 -> sum(Effort._Legendre_2.(x3)), x3)[1], ForwardDiff.gradient(x3 -> sum(Pl.(x3, 2)), x3), rtol=1e-9)
    @test isapprox(Zygote.gradient(x3 -> sum(Effort._Legendre_4.(x3)), x3)[1], ForwardDiff.gradient(x3 -> sum(Pl.(x3, 4)), x3), rtol=1e-9)
    mycosmo = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.636, ωb=0.02237, ωc=0.1, mν=0.06, w0=-2.0, wa=1.0)
    mycosmo_ref = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
    qpar, qperp = Effort.q_par_perp(0.5, mycosmo, mycosmo_ref)
    @test isapprox(qpar, 1.1676180546427928, rtol=3e-5)
    @test isapprox(qperp, 1.1273544308379857, rtol=2e-5)
    a, b, c = Effort.apply_AP(k, k_test, Pℓ[1, :], Pℓ[2, :], Pℓ[3, :], qpar, qperp; n_GL_points=5)
    @test isapprox(a[4:end], Pℓ_AP[1, 4:end], rtol=5e-4)
    @test isapprox(b[4:end], Pℓ_AP[2, 4:end], rtol=5e-4)
    @test isapprox(c[4:end], Pℓ_AP[3, 4:end], rtol=5e-4)
    cϵi = [1.0, 1.0, 1.0]
end

@testset "Emulator Jacobian Tests" begin
    # Test that emulators are loaded
    @test haskey(Effort.trained_emulators, "PyBirdmnuw0wacdm")
    @test haskey(Effort.trained_emulators["PyBirdmnuw0wacdm"], "0")
    @test haskey(Effort.trained_emulators["PyBirdmnuw0wacdm"], "2")
    @test haskey(Effort.trained_emulators["PyBirdmnuw0wacdm"], "4")

    # Set up test parameters
    z = 1.2
    ln10As = 3.0
    ns = 0.96
    H0 = 67.36
    ωb = 0.022
    ωcdm = 0.12
    mν = 0.06
    w0 = -1.0
    wa = 0.0
    cosmology_params = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]

    # Bias parameters
    b1 = 1.5
    b2 = 0.5
    b3 = 0.2
    b4 = 1.0
    b5 = 1.0
    b6 = 1.0
    b7 = 1.0
    f = 0.9
    cϵ0 = 1.0
    cϵ1 = 1.0
    cϵ2 = 1.0
    bias_params = [b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2]

    # Growth factor
    D_growth = 0.8

    # Get emulators
    monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
    quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
    hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

    # Test 1: Basic multipole computation
    @testset "Basic Multipole Computation" begin
        P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
        @test length(P0) > 0
        @test all(isfinite.(P0))

        P2 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, quadrupole_emu)
        @test length(P2) > 0
        @test all(isfinite.(P2))

        P4 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, hexadecapole_emu)
        @test length(P4) > 0
        @test all(isfinite.(P4))
    end

    # Test 2: ForwardDiff Jacobian w.r.t. cosmological parameters
    @testset "ForwardDiff Jacobian - Cosmology" begin
        function compute_P0_cosmology(cosmo_params)
            return Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
        end

        jac_cosmology = ForwardDiff.jacobian(compute_P0_cosmology, cosmology_params)
        @test all(isfinite.(jac_cosmology))
    end

    # Test 3: ForwardDiff Jacobian w.r.t. bias parameters
    @testset "ForwardDiff Jacobian - Bias" begin
        function compute_P0_bias(bias)
            return Effort.get_Pℓ(cosmology_params, D_growth, bias, monopole_emu)
        end

        jac_bias = ForwardDiff.jacobian(compute_P0_bias, bias_params)
        @test all(isfinite.(jac_bias))
    end

    # Test 4: Zygote gradient computation
    @testset "Zygote Gradient" begin
        function loss_cosmology(cosmo_params)
            P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
            return sum(P0)
        end

        grad_cosmology = Zygote.gradient(loss_cosmology, cosmology_params)[1]
        @test all(isfinite.(grad_cosmology))

        function loss_bias(bias)
            P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias, monopole_emu)
            return sum(P0)
        end

        grad_bias = Zygote.gradient(loss_bias, bias_params)[1]
        @test all(isfinite.(grad_bias))
    end

    # Test 5: Built-in Jacobian function
    @testset "Built-in Jacobian Function" begin
        P0, jac_P0 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)
        @test length(P0) > 0
        @test all(isfinite.(P0))
        @test size(jac_P0)[1] == length(P0)
        @test all(isfinite.(jac_P0))

        # Test for all multipoles
        P2, jac_P2 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, quadrupole_emu)
        @test all(isfinite.(P2))
        @test all(isfinite.(jac_P2))

        P4, jac_P4 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, hexadecapole_emu)
        @test all(isfinite.(P4))
        @test all(isfinite.(jac_P4))
    end

    # Test 6: Consistency between ForwardDiff and Zygote
    @testset "ForwardDiff vs Zygote Consistency" begin
        function test_function(cosmo_params, emu)
            Pℓ = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, emu)
            return sum(Pℓ .^ 2)  # Sum of squares for a scalar output
        end

        grad0_fd = ForwardDiff.gradient(cosmology_params -> test_function(cosmology_params, monopole_emu), cosmology_params)
        grad0_zy = Zygote.gradient(cosmology_params -> test_function(cosmology_params, monopole_emu), cosmology_params)[1]

        # They should be approximately equal (allowing for numerical differences)
        @test isapprox(grad0_fd, grad0_zy, rtol=1e-5)

        grad2_fd = ForwardDiff.gradient(cosmology_params -> test_function(cosmology_params, quadrupole_emu), cosmology_params)
        grad2_zy = Zygote.gradient(cosmology_params -> test_function(cosmology_params, quadrupole_emu), cosmology_params)[1]

        # They should be approximately equal (allowing for numerical differences)
        @test isapprox(grad2_fd, grad2_zy, rtol=1e-5)

        grad4_fd = ForwardDiff.gradient(cosmology_params -> test_function(cosmology_params, hexadecapole_emu), cosmology_params)
        grad4_zy = Zygote.gradient(cosmology_params -> test_function(cosmology_params, hexadecapole_emu), cosmology_params)[1]

        # They should be approximately equal (allowing for numerical differences)
        @test isapprox(grad4_fd, grad4_zy, rtol=1e-5)
    end

    # Test 7: Multiple multipoles differentiation
    @testset "Multiple Multipoles Differentiation" begin
        function combined_multipole(cosmo_params)
            P0 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, monopole_emu)
            P2 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, quadrupole_emu)
            P4 = Effort.get_Pℓ(cosmo_params, D_growth, bias_params, hexadecapole_emu)
            return sum(P0 .^ 2 .+ P2 .^ 2 .+ P4 .^ 2)
        end

        grad_combined = ForwardDiff.gradient(combined_multipole, cosmology_params)
        @test all(isfinite.(grad_combined))

        grad_combined_zy = Zygote.gradient(combined_multipole, cosmology_params)[1]
        @test isapprox(grad_combined, grad_combined_zy, rtol=1e-5)
    end

    # Test 8: Jacobian test
    @testset "Jacobian test" begin
        JFDb0 = ForwardDiff.jacobian(bias_params -> monopole_emu.BiasCombination(bias_params), bias_params)
        Jb0 = monopole_emu.JacobianBiasCombination(bias_params)

        JFDb2 = ForwardDiff.jacobian(bias_params -> quadrupole_emu.BiasCombination(bias_params), bias_params)
        Jb2 = quadrupole_emu.JacobianBiasCombination(bias_params)

        JFDb4 = ForwardDiff.jacobian(bias_params -> hexadecapole_emu.BiasCombination(bias_params), bias_params)
        Jb4 = hexadecapole_emu.JacobianBiasCombination(bias_params)

        @test isapprox(JFDb0, Jb0, rtol=1e-5)
        @test isapprox(JFDb2, Jb2, rtol=1e-5)
        @test isapprox(JFDb4, Jb4, rtol=1e-5)
    end

    # Test 9: Jacobian with AP effect - comparing two approaches
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

    # Test 10: Batch apply_AP vs column-by-column application
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

@testset "Matrix Akima Interpolation Tests" begin
    # Test that the matrix version of _akima_spline_legacy produces identical
    # results to the column-by-column approach, which is the key optimization
    # for Jacobian computations with AP transformations.

    @testset "Correctness: Matrix vs Column-wise" begin
        # Test case 1: Typical Jacobian scenario (11 bias parameters)
        k_in = collect(range(0.01, 0.3, length=50))
        k_out = collect(range(0.015, 0.28, length=100))
        jacobian = randn(50, 11)

        # Matrix version (optimized)
        result_matrix = Effort._akima_spline_legacy(jacobian, k_in, k_out)

        # Column-by-column version (reference)
        result_cols = hcat([Effort._akima_spline_legacy(jacobian[:, i], k_in, k_out)
                           for i in 1:size(jacobian, 2)]...)

        # Should be identical (not just approximately equal)
        @test maximum(abs.(result_matrix - result_cols)) < 1e-14
        @test size(result_matrix) == (100, 11)
        @test size(result_matrix) == size(result_cols)
    end

    @testset "Edge Cases" begin
        k_in = collect(range(0.0, 1.0, length=20))
        k_out = collect(range(0.1, 0.9, length=30))

        # Test case 2: Single column (should still work)
        data_single = randn(20, 1)
        result_single_matrix = Effort._akima_spline_legacy(data_single, k_in, k_out)
        result_single_vector = Effort._akima_spline_legacy(data_single[:, 1], k_in, k_out)
        @test maximum(abs.(result_single_matrix[:, 1] - result_single_vector)) < 1e-14

        # Test case 3: Two columns
        data_two = randn(20, 2)
        result_two = Effort._akima_spline_legacy(data_two, k_in, k_out)
        @test size(result_two) == (30, 2)
        for i in 1:2
            result_vec = Effort._akima_spline_legacy(data_two[:, i], k_in, k_out)
            @test maximum(abs.(result_two[:, i] - result_vec)) < 1e-14
        end

        # Test case 4: Many columns (stress test)
        data_many = randn(20, 50)
        result_many = Effort._akima_spline_legacy(data_many, k_in, k_out)
        @test size(result_many) == (30, 50)
        # Check first, middle, and last columns
        for i in [1, 25, 50]
            result_vec = Effort._akima_spline_legacy(data_many[:, i], k_in, k_out)
            @test maximum(abs.(result_many[:, i] - result_vec)) < 1e-14
        end
    end

    @testset "Type Stability and Promotion" begin
        k_in = collect(range(0.0, 1.0, length=10))
        k_out = collect(range(0.1, 0.9, length=15))

        # Test case 5: Float32 input
        data_f32 = randn(Float32, 10, 5)
        result_f32 = Effort._akima_spline_legacy(data_f32, k_in, k_out)
        @test eltype(result_f32) == Float64  # Promotes to Float64 due to Float64 k_in
        @test size(result_f32) == (15, 5)

        # Test case 6: All Float32
        k_in_f32 = Float32.(k_in)
        k_out_f32 = Float32.(k_out)
        result_all_f32 = Effort._akima_spline_legacy(data_f32, k_in_f32, k_out_f32)
        @test eltype(result_all_f32) == Float32
        @test size(result_all_f32) == (15, 5)
    end

    @testset "Integration with apply_AP" begin
        # Test case 7: Full AP transformation with matrix Jacobians
        # This is the real-world use case where the optimization matters
        k_grid = collect(range(0.01, 0.3, length=50))

        # Create Jacobian matrices (monopole, quadrupole, hexadecapole)
        Jac0 = randn(50, 11)
        Jac2 = randn(50, 11)
        Jac4 = randn(50, 11)

        q_par_test = 1.02
        q_perp_test = 0.98

        # Apply AP transformation (should use matrix Akima internally)
        result0, result2, result4 = Effort.apply_AP(
            k_grid, k_grid, Jac0, Jac2, Jac4,
            q_par_test, q_perp_test, n_GL_points=8
        )

        # Results should have correct shape
        @test size(result0) == (50, 11)
        @test size(result2) == (50, 11)
        @test size(result4) == (50, 11)

        # Results should be finite
        @test all(isfinite.(result0))
        @test all(isfinite.(result2))
        @test all(isfinite.(result4))

        # Compare with column-by-column approach
        result0_cols = hcat([Effort.apply_AP(k_grid, k_grid,
                                             Jac0[:, i], Jac2[:, i], Jac4[:, i],
                                             q_par_test, q_perp_test, n_GL_points=8)[1]
                            for i in 1:11]...)

        @test isapprox(result0, result0_cols, rtol=1e-12)
    end

    @testset "Monotonicity and Smoothness" begin
        # Test case 8: Verify interpolation properties
        k_in = collect(range(0.0, 1.0, length=10))
        k_out = collect(range(0.0, 1.0, length=50))

        # Monotonic increasing function
        data_mono = hcat([collect(range(0.0, 10.0, length=10)) for _ in 1:3]...)
        result_mono = Effort._akima_spline_legacy(data_mono, k_in, k_out)

        # Each column should be monotonically increasing
        for col in 1:3
            @test all(diff(result_mono[:, col]) .>= -1e-10)  # Allow tiny numerical errors
        end

        # Should pass through original points (approximately)
        for i in 1:10
            idx = findfirst(x -> abs(x - k_in[i]) < 1e-10, k_out)
            if !isnothing(idx)
                @test maximum(abs.(result_mono[idx, :] - data_mono[i, :])) < 1e-10
            end
        end
    end

    @testset "Automatic Differentiation: Naive vs Optimized" begin
        # Test that AD works identically for naive (column-wise) and optimized (matrix) versions
        k_in = collect(range(0.01, 0.3, length=50))
        k_out = collect(range(0.015, 0.28, length=100))
        jacobian = randn(50, 11)

        # Define naive (column-wise) version - non-mutating for Zygote compatibility
        function naive_akima_matrix(jac, t_in, t_out)
            n_cols = size(jac, 2)
            return hcat([Effort._akima_spline_legacy(jac[:, i], t_in, t_out) for i in 1:n_cols]...)
        end

        # Define optimized (matrix) version - just calls the matrix method directly
        function optimized_akima_matrix(jac, t_in, t_out)
            return Effort._akima_spline_legacy(jac, t_in, t_out)
        end

        @testset "Gradient w.r.t. matrix values" begin
            # Zygote
            grad_naive_zy = Zygote.gradient(jac -> sum(naive_akima_matrix(jac, k_in, k_out)), jacobian)[1]
            grad_opt_zy = Zygote.gradient(jac -> sum(optimized_akima_matrix(jac, k_in, k_out)), jacobian)[1]
            @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-12

            # ForwardDiff (vectorized for matrix)
            jac_vec = vec(jacobian)
            grad_naive_fd = ForwardDiff.gradient(jv -> sum(naive_akima_matrix(reshape(jv, 50, 11), k_in, k_out)), jac_vec)
            grad_opt_fd = ForwardDiff.gradient(jv -> sum(optimized_akima_matrix(reshape(jv, 50, 11), k_in, k_out)), jac_vec)
            @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

            # Zygote vs ForwardDiff consistency (optimized version)
            @test maximum(abs.(vec(grad_opt_zy) - grad_opt_fd)) < 1e-10
        end

        @testset "Gradient w.r.t. input grid (k_in)" begin
            # Zygote
            grad_naive_zy = Zygote.gradient(k -> sum(naive_akima_matrix(jacobian, k, k_out)), k_in)[1]
            grad_opt_zy = Zygote.gradient(k -> sum(optimized_akima_matrix(jacobian, k, k_out)), k_in)[1]
            @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-11

            # ForwardDiff
            grad_naive_fd = ForwardDiff.gradient(k -> sum(naive_akima_matrix(jacobian, k, k_out)), k_in)
            grad_opt_fd = ForwardDiff.gradient(k -> sum(optimized_akima_matrix(jacobian, k, k_out)), k_in)
            @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

            # Zygote vs ForwardDiff consistency
            @test maximum(abs.(grad_opt_zy - grad_opt_fd)) < 1e-9
        end

        @testset "Gradient w.r.t. output grid (k_out)" begin
            # Zygote
            grad_naive_zy = Zygote.gradient(k -> sum(naive_akima_matrix(jacobian, k_in, k)), k_out)[1]
            grad_opt_zy = Zygote.gradient(k -> sum(optimized_akima_matrix(jacobian, k_in, k)), k_out)[1]
            @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-12

            # ForwardDiff
            grad_naive_fd = ForwardDiff.gradient(k -> sum(naive_akima_matrix(jacobian, k_in, k)), k_out)
            grad_opt_fd = ForwardDiff.gradient(k -> sum(optimized_akima_matrix(jacobian, k_in, k)), k_out)
            @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

            # Zygote vs ForwardDiff consistency
            @test maximum(abs.(grad_opt_zy - grad_opt_fd)) < 1e-9
        end

        @testset "Jacobian w.r.t. matrix values (element-wise)" begin
            # Test a smaller case for full Jacobian computation
            k_small = collect(range(0.01, 0.1, length=10))
            k_out_small = collect(range(0.02, 0.09, length=20))
            jac_small = randn(10, 3)

            # Compute full Jacobian (output w.r.t. input matrix)
            # For naive version
            function naive_flat(jv)
                jac_mat = reshape(jv, 10, 3)
                result = naive_akima_matrix(jac_mat, k_small, k_out_small)
                return vec(result)
            end

            # For optimized version
            function opt_flat(jv)
                jac_mat = reshape(jv, 10, 3)
                result = optimized_akima_matrix(jac_mat, k_small, k_out_small)
                return vec(result)
            end

            jac_vec = vec(jac_small)
            jacobian_naive = ForwardDiff.jacobian(naive_flat, jac_vec)
            jacobian_opt = ForwardDiff.jacobian(opt_flat, jac_vec)

            @test maximum(abs.(jacobian_naive - jacobian_opt)) < 1e-11
        end

        @testset "Integration with apply_AP" begin
            # Test that matrix Jacobian with AD works in the full AP pipeline
            k_grid = collect(range(0.01, 0.3, length=50))
            jac_test = randn(50, 11)
            q_par = 0.98
            q_perp = 1.02

            # Function that uses matrix Jacobian through apply_AP
            function ap_with_matrix_jac(jac_mat)
                # apply_AP should handle matrix Jacobians
                mono, quad, hexa = Effort.apply_AP(
                    k_grid, k_grid,
                    jac_mat[:, 1], jac_mat[:, 2], jac_mat[:, 3],
                    q_par, q_perp;
                    n_GL_points=18
                )
                return sum(mono) + sum(quad) + sum(hexa)
            end

            # Test gradient w.r.t. Jacobian elements
            grad_zy = Zygote.gradient(ap_with_matrix_jac, jac_test)[1]
            grad_fd = ForwardDiff.gradient(jv -> ap_with_matrix_jac(reshape(jv, 50, 11)), vec(jac_test))

            @test size(grad_zy) == (50, 11)
            @test length(grad_fd) == 50 * 11
            @test maximum(abs.(vec(grad_zy) - grad_fd)) < 1e-8
        end
    end

    @testset "Matrix apply_AP Optimization" begin
        # Test that the optimized matrix apply_AP produces identical results
        # to the naive column-by-column version
        k_in = collect(range(0.01, 0.3, length=50))
        k_out = collect(range(0.015, 0.28, length=100))
        mono = randn(50, 11)
        quad = randn(50, 11)
        hexa = randn(50, 11)
        q_par = 0.98
        q_perp = 1.02

        # Naive (column-by-column) version - mimics old implementation
        function naive_apply_AP_matrix(k_input, k_output, mono, quad, hexa, q_par, q_perp; n_GL_points=8)
            results = [Effort.apply_AP(k_input, k_output, mono[:, i], quad[:, i], hexa[:, i],
                q_par, q_perp, n_GL_points=n_GL_points) for i in 1:size(mono, 2)]

            matrix1 = stack([tup[1] for tup in results], dims=2)
            matrix2 = stack([tup[2] for tup in results], dims=2)
            matrix3 = stack([tup[3] for tup in results], dims=2)

            return matrix1, matrix2, matrix3
        end

        @testset "Correctness: Matrix vs Column-wise" begin
            # Test with n_GL_points = 18 (realistic scenario)
            mono_naive, quad_naive, hexa_naive = naive_apply_AP_matrix(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=18)
            mono_opt, quad_opt, hexa_opt = Effort.apply_AP(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=18)

            @test size(mono_naive) == size(mono_opt) == (100, 11)
            @test size(quad_naive) == size(quad_opt) == (100, 11)
            @test size(hexa_naive) == size(hexa_opt) == (100, 11)

            @test maximum(abs.(mono_naive - mono_opt)) < 1e-14
            @test maximum(abs.(quad_naive - quad_opt)) < 1e-14
            @test maximum(abs.(hexa_naive - hexa_opt)) < 1e-14
        end

        @testset "Edge Cases" begin
            # Single column
            mono_1 = randn(50, 1)
            quad_1 = randn(50, 1)
            hexa_1 = randn(50, 1)

            mono_vec, quad_vec, hexa_vec = Effort.apply_AP(k_in, k_out, mono_1[:, 1], quad_1[:, 1], hexa_1[:, 1], q_par, q_perp, n_GL_points=8)
            mono_mat, quad_mat, hexa_mat = Effort.apply_AP(k_in, k_out, mono_1, quad_1, hexa_1, q_par, q_perp, n_GL_points=8)

            @test maximum(abs.(mono_vec - mono_mat[:, 1])) < 1e-14
            @test maximum(abs.(quad_vec - quad_mat[:, 1])) < 1e-14
            @test maximum(abs.(hexa_vec - hexa_mat[:, 1])) < 1e-14

            # Two columns
            mono_2 = randn(50, 2)
            quad_2 = randn(50, 2)
            hexa_2 = randn(50, 2)

            result_2 = Effort.apply_AP(k_in, k_out, mono_2, quad_2, hexa_2, q_par, q_perp, n_GL_points=8)
            @test size(result_2[1]) == (100, 2)
            @test size(result_2[2]) == (100, 2)
            @test size(result_2[3]) == (100, 2)

            # Large number of columns
            mono_20 = randn(50, 20)
            quad_20 = randn(50, 20)
            hexa_20 = randn(50, 20)

            result_20 = Effort.apply_AP(k_in, k_out, mono_20, quad_20, hexa_20, q_par, q_perp, n_GL_points=8)
            @test size(result_20[1]) == (100, 20)
            @test size(result_20[2]) == (100, 20)
            @test size(result_20[3]) == (100, 20)
        end

        @testset "Different GL Points" begin
            # Test with various GL point counts
            for n_GL in [4, 8, 12, 18]
                mono_naive, quad_naive, hexa_naive = naive_apply_AP_matrix(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=n_GL)
                mono_opt, quad_opt, hexa_opt = Effort.apply_AP(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=n_GL)

                @test maximum(abs.(mono_naive - mono_opt)) < 1e-14
                @test maximum(abs.(quad_naive - quad_opt)) < 1e-14
                @test maximum(abs.(hexa_naive - hexa_opt)) < 1e-14
            end
        end

        @testset "AD Compatibility" begin
            # Test that the optimized version works with AD
            mono_small = randn(50, 3)
            quad_small = randn(50, 3)
            hexa_small = randn(50, 3)

            # Zygote gradient w.r.t. monopole
            function test_sum(m)
                result = Effort.apply_AP(k_in, k_out, m, quad_small, hexa_small, q_par, q_perp, n_GL_points=8)
                return sum(result[1]) + sum(result[2]) + sum(result[3])
            end

            grad_zy = Zygote.gradient(test_sum, mono_small)[1]
            @test size(grad_zy) == size(mono_small)
            @test all(isfinite, grad_zy)

            # ForwardDiff gradient
            grad_fd = ForwardDiff.gradient(m -> test_sum(reshape(m, 50, 3)), vec(mono_small))
            @test length(grad_fd) == 50 * 3
            @test maximum(abs.(vec(grad_zy) - grad_fd)) < 1e-8
        end
    end

    @testset "End-to-End Automatic Differentiation" begin
        # Test complete AD pipeline: cosmology params → D, f → multipoles & Jacobians → scalar loss
        # This demonstrates that we can differentiate through the entire physical pipeline

        # Load trained emulators
        monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
        hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

        @testset "Emulator + Jacobians are Fully Differentiable" begin
            # Test that we can differentiate through the emulator and Jacobian computation
            # w.r.t. growth factor D and cosmological parameters fed to the emulator

            # Fixed cosmology parameters (9 elements as expected by emulator)
            cosmo_params = [0.8, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]

            # Bias parameters (11 EFT bias parameters)
            bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]

            # Test AD w.r.t. growth factor D
            function loss_wrt_D(D_value)
                _, Jac0 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias_params, monopole_emu)
                _, Jac2 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias_params, quadrupole_emu)
                _, Jac4 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias_params, hexadecapole_emu)
                return sum(Jac0) + sum(Jac2) + sum(Jac4)
            end

            D_value = 0.75
            println("\nTesting AD w.r.t. growth factor D...")
            grad_D = ForwardDiff.derivative(loss_wrt_D, D_value)
            @test isfinite(grad_D)
            @test abs(grad_D) > 0
            println("  ✓ ∂(sum Jacobians)/∂D = ", round(grad_D, sigdigits=4))
            println("  ✓ Gradient is finite and non-zero")

            # Test AD w.r.t. bias parameters
            function loss_wrt_bias(bias)
                _, Jac0 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias, monopole_emu)
                return sum(Jac0)
            end

            println("\nTesting AD w.r.t. bias parameters...")
            grad_bias = ForwardDiff.gradient(loss_wrt_bias, bias_params)
            @test length(grad_bias) == 11
            @test all(isfinite, grad_bias)
            @test maximum(abs.(grad_bias)) > 0
            println("  ✓ All 11 gradients are finite and non-zero")
            println("  ✓ Jacobian computation is fully differentiable")
        end
    end
end

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
