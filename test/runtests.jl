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
a, Ωγ0, Ωc0, Ωb0, mν, h, w0, wa = [1., 1e-5, 0.25, 0.05, 0., 0.67, -1, 0.]
z = [0.4, 0.5]
ΩM = Ωc0 + Ωb0
emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

effort_emu = Effort.P11Emulator(TrainedEmulator = emu, kgrid=k_test, InMinMax = inminmax,
                                OutMinMax = outminmax)

x = [Ωc0, Ωb0, h]

function pippo(z, x)
    Ωc0, Ωb0, h = x
    sum(Effort._D_z(z, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.))
end

@testset "Effort tests" begin
    @test isapprox(Effort._H_a(a, Ωγ0, Ωc0, Ωb0, mν, h, w0, wa), h*100)
    @test isapprox(Effort._E_a(a, Ωc0, Ωb0, h), 1.)
    @test isapprox(Effort._D_z_old(z, Ωc0, Ωb0, h), Effort._D_z(z, Ωc0, Ωb0, h), rtol=1e-9)
    @test isapprox(Effort._f_z_old(0.4, Ωc0, Ωb0, h), Effort._f_z(0.4, Ωc0, Ωb0, h)[1], rtol=1e-9)
    @test isapprox(Zygote.gradient(x->pippo(z, x), x)[1], ForwardDiff.gradient(x->pippo(z, x), x), rtol=1e-5)
    @test isapprox(FiniteDiff.finite_difference_gradient(x->pippo(z, x), x), ForwardDiff.gradient(x->pippo(z, x), x), rtol=1e-5)
end
