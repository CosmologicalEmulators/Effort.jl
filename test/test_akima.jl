using Test
using Effort
include("test_helpers.jl")

@testset "Akima spline tests" begin
    @test isapprox(Effort._quadratic_spline(y, x1, x2), di_spline(y, x1, x2), rtol=1e-9)
    @test isapprox(ForwardDiff.gradient(y -> sum(Effort._akima_spline_legacy(y, x1, x2)), y), Zygote.gradient(y -> sum(Effort._akima_spline_legacy(y, x1, x2)), y)[1], rtol=1e-9)
    @test isapprox(ForwardDiff.gradient(x1 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x1), Zygote.gradient(x1 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x1)[1], rtol=1e-9)
    @test isapprox(ForwardDiff.gradient(x2 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x2), Zygote.gradient(x2 -> sum(Effort._akima_spline_legacy(y, x1, x2)), x2)[1], rtol=1e-9)
end