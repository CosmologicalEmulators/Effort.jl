using Test
using NPZ
using SimpleChains
using Static
using Effort
using ForwardDiff
using Zygote
using FiniteDiff
using SciMLSensitivity

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
a, Ωγ0, Ωm0, mν, h, w0, wa = [1., 1e-5, 0.3, 0.06, 0.67, -1.1, 0.2]
z = [0.4, 0.5]

emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

effort_emu = Effort.P11Emulator(TrainedEmulator = emu, kgrid=k_test, InMinMax = inminmax,
                                OutMinMax = outminmax)

x = [Ωm0, h, mν, w0, wa]

function D_z_x(z, x)
    Ωm0, h, mν, w0, wa = x
    sum(Effort._D_z(z, Ωm0, h; mν =mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωm0, h, mν, w0, wa = x
    sum(Effort._f_z(z, Ωm0, h; mν =mν, w0=w0, wa=wa))
end

function r_z_x(z, x)
    Ωm0, h, mν, w0, wa = x
    sum(Effort._r_z(z, Ωm0, h; mν =mν, w0=w0, wa=wa))
end

@testset "Effort tests" begin
    @test isapprox(Effort._H_a(a, Ωγ0, Ωm0, mν, h, w0, wa), h*100)
    @test isapprox(Effort._E_a(a, Ωm0, h), 1.)
    @test isapprox(Effort._D_z_old(z, Ωm0, h), Effort._D_z(z, Ωm0, h), rtol=1e-9)
    @test isapprox(Effort._f_z_old(0.4, Ωm0, h), Effort._f_z(0.4, Ωm0, h)[1], rtol=1e-9)
    @test isapprox(Zygote.gradient(x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    @test isapprox(FiniteDiff.finite_difference_gradient(x->D_z_x(z, x), x), ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    @test isapprox(Zygote.gradient(x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-5)
    @test isapprox(FiniteDiff.finite_difference_gradient(x->f_z_x(z, x), x), ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-4)
    @test isapprox(FiniteDiff.finite_difference_gradient(x->r_z_x(z, x), x), ForwardDiff.gradient(x->r_z_x(z, x), x), rtol=1e-4)
end
