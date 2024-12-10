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

k_test = Array(LinRange(0,200, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6,2)
outminmax = rand(40,2)
a, Ωcb0, mν, h, w0, wa = [1., 0.3, 0.06, 0.67, -1.1, 0.2]
z = Array(LinRange(0., 3., 100))

emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

postprocessing = (input, output, D, Pkemu) -> output

effort_emu = Effort.P11Emulator(TrainedEmulator = emu, kgrid=k_test, InMinMax = inminmax,
                                OutMinMax = outminmax, Postprocessing = postprocessing)

x = [Ωcb0, h, mν, w0, wa]

n = 64
x1 = vcat([0.], sort(rand(n-2)), [1.])
x2 = 2 .* vcat([0.], sort(rand(n-2)), [1.])
y = rand(n)

W = rand(2, 20, 3, 10)
v = rand(20, 10)

function di_spline(y,x,xn)
    spline = QuadraticSpline(y,x, extrapolate = true)
    return spline.(xn)
end

function D_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function r_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._r_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function r_z_check_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._r_z_check(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

myx = Array(LinRange(0., 1., 100))
monotest = sin.(myx)
quadtest = 0.5.*cos.(myx)
hexatest = 0.1.*cos.(2 .* myx)
q_par = 1.1
q_perp = 0.9

x3 = Array(LinRange(-1., 1., 100))

@testset "Effort tests" begin
    @test isapprox(Effort._H_a(a, Ωcb0, mν, h, w0, wa), h*100)
    @test isapprox(Effort._E_a(a, Ωcb0, h), 1.)
    @test isapprox(Effort._D_z_old(z, Ωcb0, h), Effort._D_z(z, Ωcb0, h), rtol=1e-9)
    @test isapprox(Effort._f_z_old(0.4, Ωcb0, h), Effort._f_z(0.4, Ωcb0, h)[1], rtol=1e-9)
    @test isapprox(Zygote.gradient(x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    @test isapprox(grad(central_fdm(5,1), x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    @test isapprox(Zygote.gradient(x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-5)
    @test isapprox(grad(central_fdm(5,1), x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-4)
    @test isapprox(grad(central_fdm(5,1), x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-7)
    @test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-6)
    @test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], Zygote.gradient(x->r_z_check_x(3., x), x)[1], rtol=1e-7)
    @test isapprox(Effort._r_z(3., Ωcb0, h; mν =mν, w0=w0, wa=wa), Effort._r_z_check(3., Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-6)
    @test isapprox(Effort._quadratic_spline(y, x1, x2), di_spline(y, x1, x2), rtol=1e-9)
    #@test isapprox(ForwardDiff.gradient(y->sum(Effort._quadratic_spline(y,x1,x2)), y), Zygote.gradient(y->sum(Effort._quadratic_spline(y,x1,x2)), y)[1], rtol=1e-6)
    #@test isapprox(ForwardDiff.gradient(x1->sum(Effort._quadratic_spline(y,x1,x2)), x1), Zygote.gradient(x1->sum(Effort._quadratic_spline(y,x1,x2)), x1)[1], rtol=1e-6)
    #@test isapprox(ForwardDiff.gradient(x2->sum(Effort._quadratic_spline(y,x1,x2)), x2), Zygote.gradient(x2->sum(Effort._quadratic_spline(y,x1,x2)), x2)[1], rtol=1e-6)
    @test isapprox(grad(central_fdm(5,1), v->sum(Effort.window_convolution(W, v)), v)[1], Zygote.gradient(v->sum(Effort.window_convolution(W, v)), v)[1], rtol=1e-6)
    @test isapprox(grad(central_fdm(5,1), W->sum(Effort.window_convolution(W, v)), W)[1], Zygote.gradient(W->sum(Effort.window_convolution(W, v)), W)[1], rtol=1e-6)
    @test isapprox(Effort.apply_AP(myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points = 8), Effort.apply_AP_check(myx, monotest, quadtest, hexatest, q_par, q_perp), rtol=1e-3)
    @test isapprox(Effort.apply_AP(myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points = 18), Effort.apply_AP_check(myx, monotest, quadtest, hexatest, q_par, q_perp), rtol=1e-4)
    @test isapprox(Effort.apply_AP(myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points = 72), Effort.apply_AP_check(myx, monotest, quadtest, hexatest, q_par, q_perp), rtol=1e-3)
    @test isapprox(Effort.apply_AP(myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points = 126), Effort.apply_AP_check(myx, monotest, quadtest, hexatest, q_par, q_perp), rtol=1e-3)
    @test isapprox(Zygote.gradient(x3->sum(x3.*Effort._legendre_0.(x3)), x3)[1], ForwardDiff.gradient(x3->sum(x3.*Pl.(x3, 0)), x3), rtol=1e-9)
    @test isapprox(Zygote.gradient(x3->sum(Effort._legendre_2.(x3)), x3)[1], ForwardDiff.gradient(x3->sum(Pl.(x3, 2)), x3), rtol=1e-9)
    @test isapprox(Zygote.gradient(x3->sum(Effort._legendre_4.(x3)), x3)[1], ForwardDiff.gradient(x3->sum(Pl.(x3, 4)), x3), rtol=1e-9)
    #@test isapprox(Zygote.gradient(myx->sum(Effort.apply_AP(myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points = 8)), myx)[1], ForwardDiff.gradient(myx->sum(Effort.apply_AP(myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points = 8)), myx))
    #@test isapprox(Zygote.gradient(myx->sum(Effort.apply_AP(myx, monotest, quadtest, q_par, q_perp; n_GL_points = 8)), myx)[1], ForwardDiff.gradient(myx->sum(Effort.apply_AP(myx, monotest, quadtest, q_par, q_perp; n_GL_points = 8)), myx))
end
