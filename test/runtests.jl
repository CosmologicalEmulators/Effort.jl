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

function D_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

function r_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

function r_z_check_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._r_z_check(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
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

@testset "Background" begin
    @test isapprox(Effort._get_y(0.0, 1.0), 0.0)
    @test isapprox(Effort._dFdy(0.0), 0.0)
    @test isapprox(Effort._ΩνE2(1.0, 1e-4, 1.0) * 3, Effort._ΩνE2(1.0, 1e-4, ones(3)))
    @test isapprox(Effort._dΩνE2da(1.0, 1e-4, 1.0) * 3, Effort._dΩνE2da(1.0, 1e-4, ones(3)))
    @test isapprox(Effort._ρDE_z(0.0, -1.0, 1.0), 1.0)
    @test isapprox(Effort._E_a(1.0, Ωcb0, h), 1.0)
    @test isapprox(Effort._E_a(1.0, mycosmo), 1.0)
    @test isapprox(Effort._E_z(0.0, Ωcb0, h), 1.0)
    @test isapprox(Effort._E_z(0.0, Ωcb0, h), Effort._E_a(1.0, Ωcb0, h))
    @test isapprox(Effort._Ωma(1.0, Ωcb0, h), Ωcb0)
    @test isapprox(Effort._Ωma(1.0, mycosmo), (0.02237 + 0.1) / 0.636^2)
    @test isapprox(Effort._r̃_z(0.0, mycosmo), 0.0)
    @test isapprox(Effort._r_z(0.0, mycosmo), 0.0)
    @test isapprox(grad(central_fdm(5, 1), x -> r_z_x(3.0, x), x)[1], ForwardDiff.gradient(x -> r_z_x(3.0, x), x), rtol=1e-7)
    @test isapprox(Zygote.gradient(x -> r_z_x(3.0, x), x)[1], ForwardDiff.gradient(x -> r_z_x(3.0, x), x), rtol=1e-6)
    @test isapprox(Zygote.gradient(x -> r_z_x(3.0, x), x)[1], Zygote.gradient(x -> r_z_check_x(3.0, x), x)[1], rtol=1e-7)
    @test isapprox(Effort._r_z(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), Effort._r_z_check(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-6)
    @test isapprox(Effort._r_z(10.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 10161.232807937273, rtol=2e-4) #number from CLASS
    @test isapprox(Effort._dA_z(0.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), 0.0, rtol=1e-6)
    @test isapprox(Effort._dA_z(0.0, mycosmo), 0.0)
    @test isapprox(Zygote.gradient(x -> D_z_x(z, x), x)[1], ForwardDiff.gradient(x -> D_z_x(z, x), x), rtol=1e-5)
    @test isapprox(grad(central_fdm(5, 1), x -> D_z_x(z, x), x)[1], ForwardDiff.gradient(x -> D_z_x(z, x), x), rtol=1e-3)
    @test isapprox(Zygote.gradient(x -> f_z_x(z, x), x)[1], ForwardDiff.gradient(x -> f_z_x(z, x), x), rtol=1e-5)
    @test isapprox(grad(central_fdm(5, 1), x -> f_z_x(z, x), x)[1], ForwardDiff.gradient(x -> f_z_x(z, x), x), rtol=1e-4)
    @test isapprox(Effort._D_z(1.0, mycosmo), Effort._D_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0))
    @test isapprox(Effort._f_z(1.0, mycosmo), Effort._f_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0))
    @test Effort._f_z(1.0, mycosmo) == Effort._f_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
end

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
    @test isapprox(Effort._f_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 0.5336534168444999, rtol=2e-5)
    @test isapprox((Effort._D_z(1.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7) / Effort._D_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7)), 0.5713231772620894, rtol=4e-5)
    D, f = Effort._D_f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    @test isapprox(D, Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
    @test isapprox(f, Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
    @test isapprox([Effort._f_z(myz, Ωcb0, h; mν=mν, w0=w0, wa=wa) for myz in z], Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-10)
    @test isapprox([Effort._D_z(myz, Ωcb0, h; mν=mν, w0=w0, wa=wa) for myz in z], Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-10)
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
        Jb0 = monopole_emu.BiasCombination(bias_params)

        JFDb2 = ForwardDiff.jacobian(bias_params -> quadrupole_emu.BiasCombination(bias_params), bias_params)
        Jb2 = quadrupole_emu.BiasCombination(bias_params)

        JFDb4 = ForwardDiff.jacobian(bias_params -> hexadecapole_emu.BiasCombination(bias_params), bias_params)
        Jb4 = hexadecapole_emu.BiasCombination(bias_params)

        @test isapprox(JFDb0, Jb0, rtol=1e-5)
        @test isapprox(JFDb2, Jb2, rtol=1e-5)
        @test isapprox(JFDb4, Jb4, rtol=1e-5)
    end
end
