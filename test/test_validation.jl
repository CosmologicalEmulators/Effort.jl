using Test
using Effort
include("test_helpers.jl")

@testset "Validation and accuracy tests" begin
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
    
    @test isapprox(Effort._f_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7), 0.5336534168444999, rtol=2e-5)
    @test isapprox((Effort._D_z(1.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7) / Effort._D_z(0.0, 0.14 / 0.67^2, 0.67; mν=0.4, w0=-1.9, wa=0.7)), 0.5713231772620894, rtol=4e-5)
    
    D, f = Effort._D_f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    @test isapprox(D, Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
    @test isapprox(f, Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
    @test isapprox([Effort._f_z(myz, Ωcb0, h; mν=mν, w0=w0, wa=wa) for myz in z], Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-10)
    @test isapprox([Effort._D_z(myz, Ωcb0, h; mν=mν, w0=w0, wa=wa) for myz in z], Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa), rtol=1e-10)
    
    mycosmo_ref = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
    qpar, qperp = Effort.q_par_perp(0.5, mycosmo, mycosmo_ref)
    @test isapprox(qpar, 1.1676180546427928, rtol=3e-5)
    @test isapprox(qperp, 1.1273544308379857, rtol=2e-5)
    a, b, c = Effort.apply_AP(k, k_test_data, Pℓ[1, :], Pℓ[2, :], Pℓ[3, :], qpar, qperp; n_GL_points=5)
    @test isapprox(a[4:end], Pℓ_AP[1, 4:end], rtol=5e-4)
    @test isapprox(b[4:end], Pℓ_AP[2, 4:end], rtol=5e-4)
    @test isapprox(c[4:end], Pℓ_AP[3, 4:end], rtol=5e-4)
end