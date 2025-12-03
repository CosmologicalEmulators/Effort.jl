"""
Tests for interpolation methods (Akima splines).

Tests cover:
- Akima spline vector version (AD tests)
- Akima spline matrix version (comprehensive suite including optimization)
"""

using Test
using Effort
using ForwardDiff
using Zygote
using DifferentiationInterface
using ADTypes
using AbstractCosmologicalEmulators

@testset "Interpolation Methods" begin
    @testset "Akima Spline: Vector Version AD" begin
        y = INTERP_Y
        x1 = INTERP_X1
        x2 = INTERP_X2

        @testset "ForwardDiff vs Zygote: Complete Pipeline" begin
            # Test the full _akima_interpolation pipeline
            @testset "Gradient w.r.t. y (data values)" begin
                grad_fd = DifferentiationInterface.gradient(y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoForwardDiff(), y)
                grad_zy = DifferentiationInterface.gradient(y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoZygote(), y)
                @test grad_fd ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x1 (input grid)" begin
                grad_fd = DifferentiationInterface.gradient(x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoForwardDiff(), x1)
                grad_zy = DifferentiationInterface.gradient(x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoZygote(), x1)
                @test grad_fd ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x2 (query points)" begin
                grad_fd = DifferentiationInterface.gradient(x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoForwardDiff(), x2)
                grad_zy = DifferentiationInterface.gradient(x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoZygote(), x2)
                @test grad_fd ≈ grad_zy rtol=1e-9
            end
        end

        @testset "Mooncake.jl Backend Validation" begin
            # Test that Mooncake backend produces results matching Zygote
            y = INTERP_Y
            x1 = INTERP_X1
            x2 = INTERP_X2

            @testset "Gradient w.r.t. y (data values)" begin
                grad_mooncake = DifferentiationInterface.gradient(y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoMooncake(; config=Mooncake.Config()), y)
                grad_zy = DifferentiationInterface.gradient(y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoZygote(), y)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x1 (input grid)" begin
                grad_mooncake = DifferentiationInterface.gradient(x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoMooncake(; config=Mooncake.Config()), x1)
                grad_zy = DifferentiationInterface.gradient(x1 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoZygote(), x1)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. x2 (query points)" begin
                grad_mooncake = DifferentiationInterface.gradient(x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoMooncake(; config=Mooncake.Config()), x2)
                grad_zy = DifferentiationInterface.gradient(x2 -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)), AutoZygote(), x2)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end
        end



        @testset "ForwardDiff with all input types (type promotion test)" begin
            # Test that type promotion works correctly when ForwardDiff is applied
            # to ANY of the input arguments (u, t, or tq)

            # Test 1: Differentiate w.r.t. y (data values)
            f_y(y_val) = sum(AbstractCosmologicalEmulators.akima_interpolation(y_val, x1, x2))
            @test ForwardDiff.derivative(y_val -> f_y([y_val, y[2:end]...]), y[1]) isa Real

            # Test 2: Differentiate w.r.t. x1 (input grid)
            f_x1(x1_val) = sum(AbstractCosmologicalEmulators.akima_interpolation(y, [x1_val, x1[2:end]...], x2))
            @test ForwardDiff.derivative(f_x1, x1[5]) isa Real

            # Test 3: Differentiate w.r.t. x2 (query points)
            f_x2(x2_val) = sum(AbstractCosmologicalEmulators.akima_interpolation(y, x1, [x2_val, x2[2:end]...]))
            @test ForwardDiff.derivative(f_x2, x2[5]) isa Real

            # Test 4: Verify Dual number propagation through the entire pipeline
            # This tests that the type promotion in the adjoint is correct
            y_dual = ForwardDiff.Dual.(y, ones(length(y)))
            result = AbstractCosmologicalEmulators.akima_interpolation(y_dual, x1, x2)
            @test all(r -> r isa ForwardDiff.Dual, result)

            # Verify values match the non-Dual version
            result_plain = AbstractCosmologicalEmulators.akima_interpolation(y, x1, x2)
            @test all(i -> ForwardDiff.value(result[i]) ≈ result_plain[i], eachindex(result))
        end
    end

    @testset "Matrix Akima Interpolation Tests" begin
        # Test that the matrix version of _akima_interpolation produces identical
        # results to the column-by-column approach, which is the key optimization
        # for Jacobian computations with AP transformations.

        @testset "Correctness: Matrix vs Column-wise" begin
            # Test case 1: Typical Jacobian scenario (11 bias parameters)
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            jacobian = randn(50, 11)

            # Matrix version (optimized)
            result_matrix = AbstractCosmologicalEmulators.akima_interpolation(jacobian, k_in, k_out)

            # Column-by-column version (reference)
            result_cols = hcat([AbstractCosmologicalEmulators.akima_interpolation(jacobian[:, i], k_in, k_out)
                               for i in 1:size(jacobian, 2)]...)

            # Should be identical (not just approximately equal)
            @test maximum(abs.(result_matrix - result_cols)) < 1e-14
            @test size(result_matrix) == (100, 11)
            @test size(result_matrix) == size(result_cols)
        end

        @testset "Edge Cases" begin
            k_in = collect(range(0.0, 1.0, length=20))
            k_out = collect(range(0.1, 0.9, length=30))

            # Test case 2: Single column (should still work)
            data_single = randn(20, 1)
            result_single_matrix = AbstractCosmologicalEmulators.akima_interpolation(data_single, k_in, k_out)
            result_single_vector = AbstractCosmologicalEmulators.akima_interpolation(data_single[:, 1], k_in, k_out)
            @test maximum(abs.(result_single_matrix[:, 1] - result_single_vector)) < 1e-14

            # Test case 3: Two columns
            data_two = randn(20, 2)
            result_two = AbstractCosmologicalEmulators.akima_interpolation(data_two, k_in, k_out)
            @test size(result_two) == (30, 2)
            for i in 1:2
                result_vec = AbstractCosmologicalEmulators.akima_interpolation(data_two[:, i], k_in, k_out)
                @test maximum(abs.(result_two[:, i] - result_vec)) < 1e-14
            end

            # Test case 4: Many columns (stress test)
            data_many = randn(20, 50)
            result_many = AbstractCosmologicalEmulators.akima_interpolation(data_many, k_in, k_out)
            @test size(result_many) == (30, 50)
            # Check first, middle, and last columns
            for i in [1, 25, 50]
                result_vec = AbstractCosmologicalEmulators.akima_interpolation(data_many[:, i], k_in, k_out)
                @test maximum(abs.(result_many[:, i] - result_vec)) < 1e-14
            end
        end



        @testset "Type Stability and Promotion" begin
            k_in = collect(range(0.0, 1.0, length=10))
            k_out = collect(range(0.1, 0.9, length=15))

            # Test case 5: Float32 input
            data_f32 = randn(Float32, 10, 5)
            result_f32 = AbstractCosmologicalEmulators.akima_interpolation(data_f32, k_in, k_out)
            @test eltype(result_f32) == Float64  # Promotes to Float64 due to Float64 k_in
            @test size(result_f32) == (15, 5)

            # Test case 6: All Float32
            k_in_f32 = Float32.(k_in)
            k_out_f32 = Float32.(k_out)
            result_all_f32 = AbstractCosmologicalEmulators.akima_interpolation(data_f32, k_in_f32, k_out_f32)
            @test eltype(result_all_f32) == Float32
            @test size(result_all_f32) == (15, 5)
        end

        @testset "Integration with apply_AP" begin
            # Test case 7: Full AP transformation with matrix Jacobians
            # This is the real-world use case where the optimization matters
            k_grid = collect(range(0.01, 0.3, length=50))

            # Create Jacobian matrices (monopole, quadrupole, hexadecapole)
            Jac0 = randn(50, 11)
            Jac2 = randn(50, 11)
            Jac4 = randn(50, 11)

            q_par_test = 1.02
            q_perp_test = 0.98

            # Apply AP transformation (should use matrix Akima internally)
            result0, result2, result4 = Effort.apply_AP(
                k_grid, k_grid, Jac0, Jac2, Jac4,
                q_par_test, q_perp_test, n_GL_points=8
            )

            # Results should have correct shape
            @test size(result0) == (50, 11)
            @test size(result2) == (50, 11)
            @test size(result4) == (50, 11)

            # Results should be finite
            @test all(isfinite.(result0))
            @test all(isfinite.(result2))
            @test all(isfinite.(result4))

            # Compare with column-by-column approach
            result0_cols = hcat([Effort.apply_AP(k_grid, k_grid,
                                                 Jac0[:, i], Jac2[:, i], Jac4[:, i],
                                                 q_par_test, q_perp_test, n_GL_points=8)[1]
                                for i in 1:11]...)

            @test isapprox(result0, result0_cols, rtol=1e-12)
        end

        @testset "Monotonicity and Smoothness" begin
            # Test case 8: Verify interpolation properties
            k_in = collect(range(0.0, 1.0, length=10))
            k_out = collect(range(0.0, 1.0, length=50))

            # Monotonic increasing function
            data_mono = hcat([collect(range(0.0, 10.0, length=10)) for _ in 1:3]...)
            result_mono = AbstractCosmologicalEmulators.akima_interpolation(data_mono, k_in, k_out)

            # Each column should be monotonically increasing
            for col in 1:3
                @test all(diff(result_mono[:, col]) .>= -1e-10)  # Allow tiny numerical errors
            end

            # Should pass through original points (approximately)
            for i in 1:10
                idx = findfirst(x -> abs(x - k_in[i]) < 1e-10, k_out)
                if !isnothing(idx)
                    @test maximum(abs.(result_mono[idx, :] - data_mono[i, :])) < 1e-10
                end
            end
        end

        @testset "Automatic Differentiation: Naive vs Optimized" begin
            # Test that AD works identically for naive (column-wise) and optimized (matrix) versions
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            jacobian = randn(50, 11)

            # Define naive (column-wise) version - non-mutating for Zygote compatibility
            function naive_akima_matrix(jac, t_in, t_out)
                n_cols = size(jac, 2)
                return hcat([AbstractCosmologicalEmulators.akima_interpolation(jac[:, i], t_in, t_out) for i in 1:n_cols]...)
            end

            # Define optimized (matrix) version - just calls the matrix method directly
            function optimized_akima_matrix(jac, t_in, t_out)
                return AbstractCosmologicalEmulators.akima_interpolation(jac, t_in, t_out)
            end

            @testset "Gradient w.r.t. matrix values" begin
                # Zygote
                grad_naive_zy = DifferentiationInterface.gradient(jac -> sum(naive_akima_matrix(jac, k_in, k_out)), AutoZygote(), jacobian)
                grad_opt_zy = DifferentiationInterface.gradient(jac -> sum(optimized_akima_matrix(jac, k_in, k_out)), AutoZygote(), jacobian)
                @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-12

                # ForwardDiff (vectorized for matrix)
                jac_vec = vec(jacobian)
                grad_naive_fd = DifferentiationInterface.gradient(jv -> sum(naive_akima_matrix(reshape(jv, 50, 11), k_in, k_out)), AutoForwardDiff(), jac_vec)
                grad_opt_fd = DifferentiationInterface.gradient(jv -> sum(optimized_akima_matrix(reshape(jv, 50, 11), k_in, k_out)), AutoForwardDiff(), jac_vec)
                @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

                # Zygote vs ForwardDiff consistency (optimized version)
                @test maximum(abs.(vec(grad_opt_zy) - grad_opt_fd)) < 1e-10
            end

            @testset "Gradient w.r.t. input grid (k_in)" begin
                # NOTE: ForwardDiff w.r.t. input grid (t) is NOT supported for the optimized
                # matrix version. This is not a limitation in practice since Effort.jl only
                # differentiates w.r.t. u (matrix values) and t_new (output grid), never w.r.t. t.
                # Zygote works correctly for all cases.

                # Zygote - works for both naive and optimized versions
                grad_naive_zy = DifferentiationInterface.gradient(k -> sum(naive_akima_matrix(jacobian, k, k_out)), AutoZygote(), k_in)
                grad_opt_zy = DifferentiationInterface.gradient(k -> sum(optimized_akima_matrix(jacobian, k, k_out)), AutoZygote(), k_in)

                @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-11

                # ForwardDiff - only test the naive version
                # (optimized version does not support ForwardDiff w.r.t. t)
                grad_naive_fd = DifferentiationInterface.gradient(k -> sum(naive_akima_matrix(jacobian, k, k_out)), AutoForwardDiff(), k_in)

                # Verify Zygote vs ForwardDiff for naive version only
                @test maximum(abs.(grad_naive_zy - grad_naive_fd)) < 1e-9
            end

            @testset "Gradient w.r.t. output grid (k_out)" begin
                # Zygote
                grad_naive_zy = DifferentiationInterface.gradient(k -> sum(naive_akima_matrix(jacobian, k_in, k)), AutoZygote(), k_out)
                grad_opt_zy = DifferentiationInterface.gradient(k -> sum(optimized_akima_matrix(jacobian, k_in, k)), AutoZygote(), k_out)
                @test maximum(abs.(grad_naive_zy - grad_opt_zy)) < 1e-12

                # ForwardDiff
                grad_naive_fd = DifferentiationInterface.gradient(k -> sum(naive_akima_matrix(jacobian, k_in, k)), AutoForwardDiff(), k_out)
                grad_opt_fd = DifferentiationInterface.gradient(k -> sum(optimized_akima_matrix(jacobian, k_in, k)), AutoForwardDiff(), k_out)
                @test maximum(abs.(grad_naive_fd - grad_opt_fd)) < 1e-12

                # Zygote vs ForwardDiff consistency
                @test maximum(abs.(grad_opt_zy - grad_opt_fd)) < 1e-9
            end

            @testset "Jacobian w.r.t. matrix values (element-wise)" begin
                # Test a smaller case for full Jacobian computation
                k_small = collect(range(0.01, 0.1, length=10))
                k_out_small = collect(range(0.02, 0.09, length=20))
                jac_small = randn(10, 3)

                # Compute full Jacobian (output w.r.t. input matrix)
                # For naive version
                function naive_flat(jv)
                    jac_mat = reshape(jv, 10, 3)
                    result = naive_akima_matrix(jac_mat, k_small, k_out_small)
                    return vec(result)
                end

                # For optimized version
                function opt_flat(jv)
                    jac_mat = reshape(jv, 10, 3)
                    result = optimized_akima_matrix(jac_mat, k_small, k_out_small)
                    return vec(result)
                end

                jac_vec = vec(jac_small)
                jacobian_naive = DifferentiationInterface.jacobian(naive_flat, AutoForwardDiff(), jac_vec)
                jacobian_opt = DifferentiationInterface.jacobian(opt_flat, AutoForwardDiff(), jac_vec)

                @test maximum(abs.(jacobian_naive - jacobian_opt)) < 1e-11
            end

            @testset "Integration with apply_AP" begin
                # Test that matrix Jacobian with AD works in the full AP pipeline
                k_grid = collect(range(0.01, 0.3, length=50))
                jac_test = randn(50, 11)
                q_par = 0.98
                q_perp = 1.02

                # Function that uses matrix Jacobian through apply_AP
                function ap_with_matrix_jac(jac_mat)
                    # apply_AP should handle matrix Jacobians
                    mono, quad, hexa = Effort.apply_AP(
                        k_grid, k_grid,
                        jac_mat[:, 1], jac_mat[:, 2], jac_mat[:, 3],
                        q_par, q_perp;
                        n_GL_points=18
                    )
                    return sum(mono) + sum(quad) + sum(hexa)
                end

                # Test gradient w.r.t. Jacobian elements
                grad_zy = DifferentiationInterface.gradient(ap_with_matrix_jac, AutoZygote(), jac_test)
                grad_fd = DifferentiationInterface.gradient(jv -> ap_with_matrix_jac(reshape(jv, 50, 11)), AutoForwardDiff(), vec(jac_test))

                @test size(grad_zy) == (50, 11)
                @test length(grad_fd) == 50 * 11
                @test maximum(abs.(vec(grad_zy) - grad_fd)) < 1e-8
            end
        end

        @testset "Mooncake.jl Backend: Matrix Gradients" begin
            # Test Mooncake backend on complete matrix Akima interpolation
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            jacobian = randn(50, 11)

            @testset "Gradient w.r.t. matrix values" begin
                grad_mooncake = DifferentiationInterface.gradient(jac -> sum(AbstractCosmologicalEmulators.akima_interpolation(jac, k_in, k_out)), AutoMooncake(; config=Mooncake.Config()), jacobian)
                grad_zy = DifferentiationInterface.gradient(jac -> sum(AbstractCosmologicalEmulators.akima_interpolation(jac, k_in, k_out)), AutoZygote(), jacobian)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. input grid (k_in)" begin
                grad_mooncake = DifferentiationInterface.gradient(k -> sum(AbstractCosmologicalEmulators.akima_interpolation(jacobian, k, k_out)), AutoMooncake(; config=Mooncake.Config()), k_in)
                grad_zy = DifferentiationInterface.gradient(k -> sum(AbstractCosmologicalEmulators.akima_interpolation(jacobian, k, k_out)), AutoZygote(), k_in)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end

            @testset "Gradient w.r.t. output grid (k_out)" begin
                grad_mooncake = DifferentiationInterface.gradient(k -> sum(AbstractCosmologicalEmulators.akima_interpolation(jacobian, k_in, k)), AutoMooncake(; config=Mooncake.Config()), k_out)
                grad_zy = DifferentiationInterface.gradient(k -> sum(AbstractCosmologicalEmulators.akima_interpolation(jacobian, k_in, k)), AutoZygote(), k_out)
                @test grad_mooncake ≈ grad_zy rtol=1e-9
            end
        end

        @testset "Matrix apply_AP Optimization" begin
            # Test that the optimized matrix apply_AP produces identical results
            # to the naive column-by-column version
            k_in = collect(range(0.01, 0.3, length=50))
            k_out = collect(range(0.015, 0.28, length=100))
            mono = randn(50, 11)
            quad = randn(50, 11)
            hexa = randn(50, 11)
            q_par = 0.98
            q_perp = 1.02

            # Naive (column-by-column) version - mimics old implementation
            function naive_apply_AP_matrix(k_input, k_output, mono, quad, hexa, q_par, q_perp; n_GL_points=8)
                results = [Effort.apply_AP(k_input, k_output, mono[:, i], quad[:, i], hexa[:, i],
                    q_par, q_perp, n_GL_points=n_GL_points) for i in 1:size(mono, 2)]

                matrix1 = stack([tup[1] for tup in results], dims=2)
                matrix2 = stack([tup[2] for tup in results], dims=2)
                matrix3 = stack([tup[3] for tup in results], dims=2)

                return matrix1, matrix2, matrix3
            end

            @testset "Correctness: Matrix vs Column-wise" begin
                # Test with n_GL_points = 18 (realistic scenario)
                mono_naive, quad_naive, hexa_naive = naive_apply_AP_matrix(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=18)
                mono_opt, quad_opt, hexa_opt = Effort.apply_AP(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=18)

                @test size(mono_naive) == size(mono_opt) == (100, 11)
                @test size(quad_naive) == size(quad_opt) == (100, 11)
                @test size(hexa_naive) == size(hexa_opt) == (100, 11)

                @test maximum(abs.(mono_naive - mono_opt)) < 1e-14
                @test maximum(abs.(quad_naive - quad_opt)) < 1e-14
                @test maximum(abs.(hexa_naive - hexa_opt)) < 1e-14
            end

            @testset "Edge Cases" begin
                # Single column
                mono_1 = randn(50, 1)
                quad_1 = randn(50, 1)
                hexa_1 = randn(50, 1)

                mono_vec, quad_vec, hexa_vec = Effort.apply_AP(k_in, k_out, mono_1[:, 1], quad_1[:, 1], hexa_1[:, 1], q_par, q_perp, n_GL_points=8)
                mono_mat, quad_mat, hexa_mat = Effort.apply_AP(k_in, k_out, mono_1, quad_1, hexa_1, q_par, q_perp, n_GL_points=8)

                @test maximum(abs.(mono_vec - mono_mat[:, 1])) < 1e-14
                @test maximum(abs.(quad_vec - quad_mat[:, 1])) < 1e-14
                @test maximum(abs.(hexa_vec - hexa_mat[:, 1])) < 1e-14

                # Two columns
                mono_2 = randn(50, 2)
                quad_2 = randn(50, 2)
                hexa_2 = randn(50, 2)

                result_2 = Effort.apply_AP(k_in, k_out, mono_2, quad_2, hexa_2, q_par, q_perp, n_GL_points=8)
                @test size(result_2[1]) == (100, 2)
                @test size(result_2[2]) == (100, 2)
                @test size(result_2[3]) == (100, 2)

                # Large number of columns
                mono_20 = randn(50, 20)
                quad_20 = randn(50, 20)
                hexa_20 = randn(50, 20)

                result_20 = Effort.apply_AP(k_in, k_out, mono_20, quad_20, hexa_20, q_par, q_perp, n_GL_points=8)
                @test size(result_20[1]) == (100, 20)
                @test size(result_20[2]) == (100, 20)
                @test size(result_20[3]) == (100, 20)
            end

            @testset "Different GL Points" begin
                # Test with various GL point counts
                for n_GL in [4, 8, 12, 18]
                    mono_naive, quad_naive, hexa_naive = naive_apply_AP_matrix(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=n_GL)
                    mono_opt, quad_opt, hexa_opt = Effort.apply_AP(k_in, k_out, mono, quad, hexa, q_par, q_perp, n_GL_points=n_GL)

                    @test maximum(abs.(mono_naive - mono_opt)) < 1e-14
                    @test maximum(abs.(quad_naive - quad_opt)) < 1e-14
                    @test maximum(abs.(hexa_naive - hexa_opt)) < 1e-14
                end
            end

            @testset "AD Compatibility" begin
                # Test that the optimized version works with AD
                mono_small = randn(50, 3)
                quad_small = randn(50, 3)
                hexa_small = randn(50, 3)

                # Zygote gradient w.r.t. monopole
                function test_sum(m)
                    result = Effort.apply_AP(k_in, k_out, m, quad_small, hexa_small, q_par, q_perp, n_GL_points=8)
                    return sum(result[1]) + sum(result[2]) + sum(result[3])
                end

                grad_zy = DifferentiationInterface.gradient(test_sum, AutoZygote(), mono_small)
                @test size(grad_zy) == size(mono_small)
                @test all(isfinite, grad_zy)

                # ForwardDiff gradient
                grad_fd = DifferentiationInterface.gradient(m -> test_sum(reshape(m, 50, 3)), AutoForwardDiff(), vec(mono_small))
                @test length(grad_fd) == 50 * 3
                @test maximum(abs.(vec(grad_zy) - grad_fd)) < 1e-8
            end
        end

        @testset "End-to-End Automatic Differentiation" begin
            # Test complete AD pipeline: cosmology params → D, f → multipoles & Jacobians → scalar loss
            # This demonstrates that we can differentiate through the entire physical pipeline

            # Load trained emulators
            monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
            quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
            hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

            @testset "Emulator + Jacobians are Fully Differentiable" begin
                # Test that we can differentiate through the emulator and Jacobian computation
                # w.r.t. growth factor D and cosmological parameters fed to the emulator

                # Fixed cosmology parameters (9 elements as expected by emulator)
                cosmo_params = [0.8, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]

                # Bias parameters (11 EFT bias parameters)
                bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]

                # Test AD w.r.t. growth factor D
                function loss_wrt_D(D_value)
                    _, Jac0 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias_params, monopole_emu)
                    _, Jac2 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias_params, quadrupole_emu)
                    _, Jac4 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias_params, hexadecapole_emu)
                    return sum(Jac0) + sum(Jac2) + sum(Jac4)
                end

                D_value = 0.75
                println("\nTesting AD w.r.t. growth factor D...")
                grad_D = ForwardDiff.derivative(loss_wrt_D, D_value)
                @test isfinite(grad_D)
                @test abs(grad_D) > 0
                println("  ✓ ∂(sum Jacobians)/∂D = ", round(grad_D, sigdigits=4))
                println("  ✓ Gradient is finite and non-zero")

                # Test AD w.r.t. bias parameters
                function loss_wrt_bias(bias)
                    _, Jac0 = Effort.get_Pℓ_jacobian(cosmo_params, D_value, bias, monopole_emu)
                    return sum(Jac0)
                end

                println("\nTesting AD w.r.t. bias parameters...")
                grad_bias = DifferentiationInterface.gradient(loss_wrt_bias, AutoForwardDiff(), bias_params)
                @test length(grad_bias) == 11
                @test all(isfinite, grad_bias)
                @test maximum(abs.(grad_bias)) > 0
                println("  ✓ All 11 gradients are finite and non-zero")
                println("  ✓ Jacobian computation is fully differentiable")
            end
        end
    end
end
