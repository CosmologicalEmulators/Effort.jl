using Test
using Effort
include("test_helpers.jl")

@testset "Gradient tests" begin
    @test isapprox(grad(central_fdm(5, 1), x -> r_z_x(3.0, x), x)[1], ForwardDiff.gradient(x -> r_z_x(3.0, x), x), rtol=1e-7)
    @test isapprox(Zygote.gradient(x -> r_z_x(3.0, x), x)[1], ForwardDiff.gradient(x -> r_z_x(3.0, x), x), rtol=1e-6)
    @test isapprox(Zygote.gradient(x -> r_z_x(3.0, x), x)[1], Zygote.gradient(x -> r_z_check_x(3.0, x), x)[1], rtol=1e-7)
    @test isapprox(Zygote.gradient(x -> D_z_x(z, x), x)[1], ForwardDiff.gradient(x -> D_z_x(z, x), x), rtol=1e-5)
    @test isapprox(grad(central_fdm(5, 1), x -> D_z_x(z, x), x)[1], ForwardDiff.gradient(x -> D_z_x(z, x), x), rtol=1e-3)
    @test isapprox(Zygote.gradient(x -> f_z_x(z, x), x)[1], ForwardDiff.gradient(x -> f_z_x(z, x), x), rtol=1e-5)
    @test isapprox(grad(central_fdm(5, 1), x -> f_z_x(z, x), x)[1], ForwardDiff.gradient(x -> f_z_x(z, x), x), rtol=1e-4)
    
    @test isapprox(FiniteDifferences.jacobian(central_fdm(5, 1), monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest)[1], Zygote.jacobian(monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest)[1], rtol=1e-3)
    @test isapprox(ForwardDiff.jacobian(monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest), Zygote.jacobian(monotest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), monotest)[1], rtol=1e-10)
    @test isapprox(ForwardDiff.jacobian(quadtest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), quadtest), Zygote.jacobian(quadtest -> sum(Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)), quadtest)[1], rtol=1e-10)
    @test isapprox(grad(central_fdm(5, 1), v -> sum(Effort.window_convolution(W, v)), v)[1], Zygote.gradient(v -> sum(Effort.window_convolution(W, v)), v)[1], rtol=1e-6)
    @test isapprox(grad(central_fdm(5, 1), W -> sum(Effort.window_convolution(W, v)), W)[1], Zygote.gradient(W -> sum(Effort.window_convolution(W, v)), W)[1], rtol=1e-6)
    
    @test isapprox(Zygote.gradient(x3 -> sum(x3 .* Effort._Legendre_0.(x3)), x3)[1], ForwardDiff.gradient(x3 -> sum(x3 .* Pl.(x3, 0)), x3), rtol=1e-9)
    @test isapprox(Zygote.gradient(x3 -> sum(Effort._Legendre_2.(x3)), x3)[1], ForwardDiff.gradient(x3 -> sum(Pl.(x3, 2)), x3), rtol=1e-9)
    @test isapprox(Zygote.gradient(x3 -> sum(Effort._Legendre_4.(x3)), x3)[1], ForwardDiff.gradient(x3 -> sum(Pl.(x3, 4)), x3), rtol=1e-9)
end