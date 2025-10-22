"""
Tests for Alcock-Paczynski (AP) effect transformations.

Tests cover:
- apply_AP with different GL point counts
- apply_AP_check (numerical integration reference)
- Automatic differentiation (ForwardDiff, Zygote, FiniteDifferences)
- Real data validation
"""

using Test
using Effort
using ForwardDiff
using Zygote
using FiniteDifferences

@testset "Alcock-Paczynski Effect" begin
    @testset "Automatic Differentiation: apply_AP" begin
        # Use test data from fixtures
        myx = AP_X
        monotest = AP_MONOPOLE
        quadtest = AP_QUADRUPOLE
        hexatest = AP_HEXADECAPOLE
        q_par = AP_Q_PAR
        q_perp = AP_Q_PERP

        @testset "Jacobian w.r.t. monopole" begin
            # FiniteDifferences vs Zygote
            jac_fd = FiniteDifferences.jacobian(
                central_fdm(5, 1),
                mono -> sum(Effort.apply_AP(myx, myx, mono, quadtest, hexatest, q_par, q_perp; n_GL_points=18)),
                monotest
            )[1]
            jac_zy = Zygote.jacobian(
                mono -> sum(Effort.apply_AP(myx, myx, mono, quadtest, hexatest, q_par, q_perp; n_GL_points=18)),
                monotest
            )[1]

            @test jac_fd ≈ jac_zy rtol=1e-3

            # ForwardDiff vs Zygote
            jac_forwarddiff = ForwardDiff.jacobian(
                mono -> sum(Effort.apply_AP(myx, myx, mono, quadtest, hexatest, q_par, q_perp; n_GL_points=18)),
                monotest
            )

            @test jac_forwarddiff ≈ jac_zy rtol=1e-10
        end

        @testset "Jacobian w.r.t. quadrupole" begin
            # ForwardDiff vs Zygote
            jac_forwarddiff = ForwardDiff.jacobian(
                quad -> sum(Effort.apply_AP(myx, myx, monotest, quad, hexatest, q_par, q_perp; n_GL_points=18)),
                quadtest
            )
            jac_zy = Zygote.jacobian(
                quad -> sum(Effort.apply_AP(myx, myx, monotest, quad, hexatest, q_par, q_perp; n_GL_points=18)),
                quadtest
            )[1]

            @test jac_forwarddiff ≈ jac_zy rtol=1e-10
        end
    end

    @testset "Different GL Point Counts: Convergence" begin
        # Test that increasing GL points improves accuracy
        myx = AP_X
        monotest = AP_MONOPOLE
        quadtest = AP_QUADRUPOLE
        hexatest = AP_HEXADECAPOLE
        q_par = AP_Q_PAR
        q_perp = AP_Q_PERP

        # Compute with different GL point counts
        a_18, b_18, c_18 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)
        a_72, b_72, c_72 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=72)
        a_126, b_126, c_126 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=126)

        # Reference: numerical integration (apply_AP_check)
        a_c, b_c, c_c = Effort.apply_AP_check(myx, myx, monotest, quadtest, hexatest, q_par, q_perp)

        @testset "GL=18 vs check" begin
            @test a_18 ≈ a_c rtol=1e-5
            @test b_18 ≈ b_c rtol=1e-5
            @test c_18 ≈ c_c rtol=1e-5
        end

        @testset "GL=72 vs check" begin
            @test a_72 ≈ a_c rtol=1e-6
            @test b_72 ≈ b_c rtol=1e-6
            @test c_72 ≈ c_c rtol=1e-6
        end

        @testset "GL=126 vs check" begin
            @test a_126 ≈ a_c rtol=1e-7
            @test b_126 ≈ b_c rtol=1e-7
            @test c_126 ≈ c_c rtol=1e-7
        end

        @testset "Convergence: More GL points = better accuracy" begin
            # Errors should decrease with more GL points
            error_18_a = maximum(abs.(a_18 - a_c))
            error_72_a = maximum(abs.(a_72 - a_c))
            error_126_a = maximum(abs.(a_126 - a_c))

            @test error_72_a < error_18_a  # 72 better than 18
            @test error_126_a < error_72_a # 126 better than 72
        end
    end

    @testset "Real Data Validation" begin
        # Test with real data from Zenodo
        real_data = load_real_data()

        # Extract data
        k = real_data.k
        k_test = real_data.k_test
        Pℓ = real_data.Pℓ
        Pℓ_AP = real_data.Pℓ_AP

        # Cosmology setup for AP parameters
        mycosmo = Effort.w0waCDMCosmology(
            ln10Aₛ=3.0, nₛ=0.96, h=0.636,
            ωb=0.02237, ωc=0.1, mν=0.06,
            w0=-2.0, wa=1.0
        )
        mycosmo_ref = Effort.w0waCDMCosmology(
            ln10Aₛ=3.0, nₛ=0.96, h=0.6736,
            ωb=0.02237, ωc=0.12, mν=0.06,
            w0=-1.0, wa=0.0
        )

        # Compute AP parameters
        qpar, qperp = Effort.q_par_perp(0.5, mycosmo, mycosmo_ref)

        # Apply AP transformation
        a, b, c = Effort.apply_AP(k, k_test, Pℓ[1, :], Pℓ[2, :], Pℓ[3, :], qpar, qperp; n_GL_points=5)

        # Compare with expected results (skip first 3 points due to edge effects)
        @test a[4:end] ≈ Pℓ_AP[1, 4:end] rtol=5e-4
        @test b[4:end] ≈ Pℓ_AP[2, 4:end] rtol=5e-4
        @test c[4:end] ≈ Pℓ_AP[3, 4:end] rtol=5e-4
    end

    @testset "apply_AP: Edge Cases and Properties" begin
        myx = AP_X
        monotest = AP_MONOPOLE
        quadtest = AP_QUADRUPOLE
        hexatest = AP_HEXADECAPOLE

        @testset "Identity: q_par = q_perp = 1" begin
            # When q parameters are 1, result should be close to original
            # (with some interpolation error)
            a, b, c = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, 1.0, 1.0; n_GL_points=18)

            # Should be approximately equal (not exact due to interpolation)
            @test a ≈ monotest atol=1e-3
            @test b ≈ quadtest atol=1e-3
            @test c ≈ hexatest atol=1e-3
        end

        @testset "Output shape and finiteness" begin
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            mono_test = randn(50)
            quad_test = randn(50)
            hexa_test = randn(50)

            a, b, c = Effort.apply_AP(k_in, k_out, mono_test, quad_test, hexa_test, 1.2, 0.9; n_GL_points=8)

            @test length(a) == length(k_out)
            @test length(b) == length(k_out)
            @test length(c) == length(k_out)

            @test all(isfinite.(a))
            @test all(isfinite.(b))
            @test all(isfinite.(c))
        end

        @testset "Different q_par and q_perp" begin
            # Test with various AP parameters
            for (q_par_test, q_perp_test) in [(1.0, 1.0), (1.1, 0.95), (0.9, 1.05), (1.2, 1.2)]
                a, b, c = Effort.apply_AP(
                    myx, myx, monotest, quadtest, hexatest,
                    q_par_test, q_perp_test;
                    n_GL_points=8
                )

                @test all(isfinite.(a))
                @test all(isfinite.(b))
                @test all(isfinite.(c))
                @test length(a) == length(myx)
                @test length(b) == length(myx)
                @test length(c) == length(myx)
            end
        end
    end

    @testset "apply_AP_check: Numerical Integration Reference" begin
        myx = AP_X
        monotest = AP_MONOPOLE
        quadtest = AP_QUADRUPOLE
        hexatest = AP_HEXADECAPOLE
        q_par = AP_Q_PAR
        q_perp = AP_Q_PERP

        @testset "Basic functionality" begin
            a, b, c = Effort.apply_AP_check(myx, myx, monotest, quadtest, hexatest, q_par, q_perp)

            @test length(a) == length(myx)
            @test length(b) == length(myx)
            @test length(c) == length(myx)

            @test all(isfinite.(a))
            @test all(isfinite.(b))
            @test all(isfinite.(c))
        end

        @testset "Consistency between calls" begin
            # Should give same result on repeated calls
            a1, b1, c1 = Effort.apply_AP_check(myx, myx, monotest, quadtest, hexatest, q_par, q_perp)
            a2, b2, c2 = Effort.apply_AP_check(myx, myx, monotest, quadtest, hexatest, q_par, q_perp)

            @test a1 ≈ a2 atol=1e-14
            @test b1 ≈ b2 atol=1e-14
            @test c1 ≈ c2 atol=1e-14
        end
    end

    @testset "Different GL Point Counts: Performance vs Accuracy" begin
        # This test characterizes the trade-off between GL points and accuracy
        myx = collect(range(0.01, 0.3, length=50))
        mono = sin.(myx)
        quad = 0.5 .* cos.(myx)
        hexa = 0.1 .* cos.(2 .* myx)
        q_par = 1.05
        q_perp = 0.95

        # Reference: high GL count
        a_ref, b_ref, c_ref = Effort.apply_AP(myx, myx, mono, quad, hexa, q_par, q_perp; n_GL_points=126)

        # Test various GL point counts
        gl_counts = [4, 8, 12, 18, 36]
        errors = Float64[]

        for n_GL in gl_counts
            a, b, c = Effort.apply_AP(myx, myx, mono, quad, hexa, q_par, q_perp; n_GL_points=n_GL)
            error = maximum(abs.(a - a_ref))
            push!(errors, error)
        end

        # Errors should generally decrease with more GL points
        # (with some fluctuations due to numerical integration)
        @test errors[end] < errors[1]  # 36 better than 4
    end
end
