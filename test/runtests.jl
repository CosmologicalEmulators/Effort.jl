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
npzwrite("emu/l.npy", k_test)
npzwrite("emu/weights.npy", weights)
npzwrite("emu/inminmax.npy", inminmax)
npzwrite("emu/outminmax.npy", outminmax)
emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

effort = Effort.P11Emulator(TrainedEmulator = emu, kgrid=k_test, InMinMax = inminmax,
                                OutMinMax = outminmax)

@testset "Effort tests" begin
    cosmo = ones(6)
    cosmo_vec = ones(6,6)
    output = Effort.get_component(cosmo,  capse_emu)
    output_vec = Effort.get_component(cosmo_vec, capse_emu)
    @test isapprox(output_vec[:,1], output)
end
