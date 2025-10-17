using Test
using Effort

@testset "Utility functions tests" begin
    # Test Legendre polynomial functions
    # L₀(x) = 1 (constant function)
    @test Effort._Legendre_0(0.5) ≈ 1.0
    @test Effort._Legendre_0(-1.0) ≈ 1.0
    @test Effort._Legendre_0(1.0) ≈ 1.0

    # L₂(x) = ½(3x² - 1)
    @test Effort._Legendre_2(0.0) ≈ -0.5     # ½(3(0)² - 1) = ½(-1) = -0.5
    @test Effort._Legendre_2(1.0) ≈ 1.0      # ½(3(1)² - 1) = ½(2) = 1.0
    @test Effort._Legendre_2(-1.0) ≈ 1.0     # ½(3(-1)² - 1) = ½(2) = 1.0
    @test Effort._Legendre_2(0.5) ≈ -0.125   # ½(3(0.5)² - 1) = ½(0.75 - 1) = ½(-0.25) = -0.125

    # L₄(x) = ⅛(35x⁴ - 30x² + 3)
    @test Effort._Legendre_4(0.0) ≈ 0.375        # ⅛(35(0)⁴ - 30(0)² + 3) = ⅛(3) = 0.375
    @test Effort._Legendre_4(1.0) ≈ 1.0          # ⅛(35(1)⁴ - 30(1)² + 3) = ⅛(8) = 1.0
    @test Effort._Legendre_4(-1.0) ≈ 1.0         # ⅛(35(-1)⁴ - 30(-1)² + 3) = ⅛(8) = 1.0
    @test Effort._Legendre_4(0.5) ≈ -0.2890625   # ⅛(35(0.5)⁴ - 30(0.5)² + 3) = ⅛(2.1875 - 7.5 + 3) = ⅛(-2.3125) = -0.2890625

    # Test transformed weights function with Gauss-Legendre quadrature
    function simple_gauss_legendre(n)
        if n == 2
            return [-1 / sqrt(3), 1 / sqrt(3)], [1.0, 1.0]
        elseif n == 3
            return [-sqrt(3 / 5), 0.0, sqrt(3 / 5)], [5 / 9, 8 / 9, 5 / 9]
        else
            error("Only n=2,3 implemented for testing")
        end
    end

    x_trans, w_trans = Effort._transformed_weights(simple_gauss_legendre, 2, 0.0, 2.0)
    @test length(x_trans) == 2
    @test length(w_trans) == 2
    @test all(x_trans .>= 0.0)
    @test all(x_trans .<= 2.0)
    @test sum(w_trans) ≈ 2.0  # Should equal interval width

    x_trans3, w_trans3 = Effort._transformed_weights(simple_gauss_legendre, 3, -1.0, 1.0)
    @test sum(w_trans3) ≈ 2.0
    @test all(x_trans3 .>= -1.0)
    @test all(x_trans3 .<= 1.0)

    # Test Akima spline legacy implementation
    t_akima = [1.0, 2.0, 3.0, 4.0, 5.0]
    u_akima = [1.0, 4.0, 9.0, 16.0, 25.0]  # x^2 values
    t_new_akima = [1.5, 2.5, 3.5, 4.5]

    result_akima = Effort._akima_spline_legacy(u_akima, t_akima, t_new_akima)
    @test length(result_akima) == 4
    @test all(result_akima .> 0.0)

    # Test single point evaluation
    single_result = Effort._akima_spline_legacy(u_akima, t_akima, 2.5)
    @test single_result > 0.0
    @test typeof(single_result) == Float64

end

@testset "Akima internal functions tests" begin
    t_test = [1.0, 2.0, 3.0, 4.0, 5.0]
    u_test = [1.0, 4.0, 9.0, 16.0, 25.0]

    # Test Akima slopes calculation
    m = Effort._akima_slopes(u_test, t_test)
    @test length(m) == length(u_test) + 3
    @test all(isfinite.(m))

    # Test Akima coefficients calculation
    b, c, d = Effort._akima_coefficients(t_test, m)
    @test length(b) == length(t_test)
    @test length(c) == length(t_test) - 1
    @test length(d) == length(t_test) - 1
    @test all(isfinite.(b))
    @test all(isfinite.(c))
    @test all(isfinite.(d))

    # Test interval finding
    idx = Effort._akima_find_interval(t_test, 2.5)
    @test idx == 2

    idx_start = Effort._akima_find_interval(t_test, 0.5)
    @test idx_start == 1

    idx_end = Effort._akima_find_interval(t_test, 5.5)
    @test idx_end == 4

    # Test evaluation at single point
    result_single = Effort._akima_eval(u_test, t_test, b, c, d, 2.5)
    @test result_single > 0.0
    @test isfinite(result_single)

    # Test evaluation at multiple points
    t_query = [1.5, 2.5, 3.5]
    result_multi = Effort._akima_eval(u_test, t_test, b, c, d, t_query)
    @test length(result_multi) == 3
    @test all(result_multi .> 0.0)
    @test all(isfinite.(result_multi))
end
