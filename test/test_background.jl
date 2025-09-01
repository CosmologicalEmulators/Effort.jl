using Test
using Effort
include("test_helpers.jl")

@testset "Background cosmology tests" begin
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
    @test isapprox(Effort._r_z(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), Effort._r_z_check(3.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-6)
    @test isapprox(Effort._r_z(10.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 10161.232807937273, rtol=2e-4)
    @test isapprox(Effort._dA_z(0.0, Ωcb0, h; mν=mν, w0=w0, wa=wa), 0.0, rtol=1e-6)
    @test isapprox(Effort._dA_z(0.0, mycosmo), 0.0)
    @test Effort._D_z(1.0, mycosmo) == Effort._D_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
    @test Effort._f_z(1.0, mycosmo) == Effort._f_z(1.0, (0.02237 + 0.1) / 0.636^2, 0.636; mν=0.06, w0=-2.0, wa=1.0)
end

@testset "Missing coverage tests" begin
    test_cosmo = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.67, ωb=0.02, ωc=0.11, mν=0.0, w0=-1.0, wa=0.0)

    @test Effort._F(0.5) > 0.0
    @test isapprox(Effort._a_z(0.0), 1.0)
    @test isapprox(Effort._a_z(1.0), 0.5)
    @test isapprox(Effort._ρDE_a(1.0, -1.0, 0.0), 1.0)
    @test isapprox(Effort._dρDEda(1.0, -1.0, 0.0), 0.0)
    @test isapprox(Effort._d̃A_z(0.0, Ωcb0, h), 0.0)
    @test isapprox(Effort._d̃A_z(0.0, mycosmo), 0.0)

    D_f = Effort._D_f_z(1.0, mycosmo)
    @test length(D_f) == 2
    @test all(D_f[1] .> 0.0)
    @test all(D_f[2] .> 0.0)
end
