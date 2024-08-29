using Test
using NPZ
using SimpleChains
using Static
using Effort
using ForwardDiff
using Zygote
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
z, a, Ωγ0, Ωc0, Ωb0, mν, h, w0, wa = [0., 1., 1e-5, 0.25, 0.05, 0., 0.67, -1, 0.]
ΩM = Ωc0 + Ωb0
emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

effort_emu = Effort.P11Emulator(TrainedEmulator = emu, kgrid=k_test, InMinMax = inminmax,
                                OutMinMax = outminmax)

x = [Ωc0, Ωb0, h]

function pippo(z, x)
    Ωc0, Ωb0, h = x
    sum(Effort.growth_solver(z, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.))
end

@testset "Effort tests" begin
    @test isapprox(Effort._H_a(a, Ωγ0, Ωc0, Ωb0, mν, h, w0, wa), h*100)
    @test isapprox(Effort._E_a(a, Ωc0, Ωb0, h), 1.)
    @test isapprox(Zygote.gradient(x->pippo(z, x), x)[1], ForwardDiff.gradient(x->pippo(z, x), x), rtol=1e-6)
end
