using Test
using NPZ
using SimpleChains
using Static
using Effort

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
emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

effort_emu = Effort.P11Emulator(TrainedEmulator = emu, kgrid=k_test, InMinMax = inminmax,
                                OutMinMax = outminmax)

@testset "Effort tests" begin
    @test isapprox(Effort._H_a(a, Ωγ0, Ωc0, Ωb0, mν, h, w0, wa), h*100)
end
