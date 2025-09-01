using Test
using Effort
include("test_helpers.jl")

@testset "Validation and accuracy tests" begin
    @test isapprox(Effort._f_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 0.5336534168444999, rtol=2e-5)
    @test isapprox((Effort._D_z(1.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7) / Effort._D_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7)), 0.5713231772620894, rtol=4e-5)
    
    D, f = Effort._D_f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    @test isapprox(D, Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
    @test isapprox(f, Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
    @test isapprox([Effort._f_z(myz, Ωcb0, h; mν=mν, w0=w0, wa=wa) for myz in z], Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-10)
    @test isapprox([Effort._D_z(myz, Ωcb0, h; mν=mν, w0=w0, wa=wa) for myz in z], Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-10)
end