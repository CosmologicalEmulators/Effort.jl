using Test
using Enzyme
using ForwardDiff
using Reactant
using Effort

@testset "Effort ExtReactant: apply_AP + window" begin
    ext_reactant = Base.get_extension(Effort, :ExtReactant)
    @test !isnothing(ext_reactant)
    Reactant.set_default_backend("cpu")

        # Deterministic synthetic setup: no ODE/background quantities anywhere.
        nk_in = 60
        nk_out = 45
        n_window = 30
        k_input = collect(range(0.01, 0.35, length=nk_in))
        k_output = collect(range(0.015, 0.30, length=nk_out))
        mono = @. exp(-k_input / 0.10)
        quad = @. 0.4 * k_input * exp(-k_input / 0.12)
        hexa = @. 0.1 * k_input^2 * exp(-k_input / 0.14)
        q_par = 1.03
        q_perp = 0.97
        W = reshape(sin.(range(0.1, 2.7, length=n_window * nk_out)), n_window, nk_out)

        @testset "Host arrays keep base apply_AP dispatch" begin
            vector_method = which(
                Effort.apply_AP,
                Tuple{
                    typeof(k_input),
                    typeof(k_output),
                    typeof(mono),
                    typeof(quad),
                    typeof(hexa),
                    typeof(q_par),
                    typeof(q_perp),
                },
            )
            @test vector_method.module === Effort

            mono_matrix = hcat(mono, 2 .* mono)
            quad_matrix = hcat(quad, 2 .* quad)
            hexa_matrix = hcat(hexa, 2 .* hexa)
            matrix_method = which(
                Effort.apply_AP,
                Tuple{
                    typeof(k_input),
                    typeof(k_output),
                    typeof(mono_matrix),
                    typeof(quad_matrix),
                    typeof(hexa_matrix),
                    typeof(q_par),
                    typeof(q_perp),
                },
            )
            @test matrix_method.module === Effort
        end

        function ap_window_outputs(mono, quad, hexa, k_input, k_output, q_par, q_perp, W)
            p0, p2, p4 = Effort.apply_AP(
                k_input,
                k_output,
                mono,
                quad,
                hexa,
                q_par,
                q_perp;
                n_GL_points=8,
                method=Effort.Cubic(),
            )
            return Effort.window_convolution(W, p0),
                Effort.window_convolution(W, p2),
                Effort.window_convolution(W, p4)
        end

        function ap_window_loss(mono, quad, hexa, k_input, k_output, q_par, q_perp, W)
            c0, c2, c4 = ap_window_outputs(mono, quad, hexa, k_input, k_output, q_par, q_perp, W)
            return sum(c0) + sum(c2) + sum(c4)
        end

        @testset "Reactant compile + forward parity" begin
            c0_ref, c2_ref, c4_ref = ap_window_outputs(mono, quad, hexa, k_input, k_output, q_par, q_perp, W)

            monoR = Reactant.to_rarray(mono)
            quadR = Reactant.to_rarray(quad)
            hexaR = Reactant.to_rarray(hexa)
            k_inputR = Reactant.to_rarray(k_input)
            k_outputR = Reactant.to_rarray(k_output)
            WR = Reactant.to_rarray(W)

            compiled_outputs = Reactant.@compile sync=true ap_window_outputs(
                monoR,
                quadR,
                hexaR,
                k_inputR,
                k_outputR,
                q_par,
                q_perp,
                WR,
            )

            c0_R, c2_R, c4_R = compiled_outputs(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)
            Reactant.synchronize(c0_R)
            Reactant.synchronize(c2_R)
            Reactant.synchronize(c4_R)

            @test Array(c0_R) ≈ c0_ref atol=1e-8 rtol=1e-8
            @test Array(c2_R) ≈ c2_ref atol=1e-8 rtol=1e-8
            @test Array(c4_R) ≈ c4_ref atol=1e-8 rtol=1e-8
        end

        @testset "Compiled Enzyme gradients wrt multipoles" begin
            grad_mono_ref = ForwardDiff.gradient(m -> ap_window_loss(m, quad, hexa, k_input, k_output, q_par, q_perp, W), mono)
            grad_quad_ref = ForwardDiff.gradient(q -> ap_window_loss(mono, q, hexa, k_input, k_output, q_par, q_perp, W), quad)
            grad_hexa_ref = ForwardDiff.gradient(h -> ap_window_loss(mono, quad, h, k_input, k_output, q_par, q_perp, W), hexa)

            grad_mono_fun(mono, quad, hexa, k_input, k_output, q_par, q_perp, W) =
                Enzyme.gradient(Reverse, ap_window_loss,
                    mono,
                    Const(quad),
                    Const(hexa),
                    Const(k_input),
                    Const(k_output),
                    Const(q_par),
                    Const(q_perp),
                    Const(W),
                )[1]

            grad_quad_fun(mono, quad, hexa, k_input, k_output, q_par, q_perp, W) =
                Enzyme.gradient(Reverse, ap_window_loss,
                    Const(mono),
                    quad,
                    Const(hexa),
                    Const(k_input),
                    Const(k_output),
                    Const(q_par),
                    Const(q_perp),
                    Const(W),
                )[2]

            grad_hexa_fun(mono, quad, hexa, k_input, k_output, q_par, q_perp, W) =
                Enzyme.gradient(Reverse, ap_window_loss,
                    Const(mono),
                    Const(quad),
                    hexa,
                    Const(k_input),
                    Const(k_output),
                    Const(q_par),
                    Const(q_perp),
                    Const(W),
                )[3]

            monoR = Reactant.to_rarray(mono)
            quadR = Reactant.to_rarray(quad)
            hexaR = Reactant.to_rarray(hexa)
            k_inputR = Reactant.to_rarray(k_input)
            k_outputR = Reactant.to_rarray(k_output)
            WR = Reactant.to_rarray(W)

            grad_mono_compiled = Reactant.@compile sync=true grad_mono_fun(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)
            grad_quad_compiled = Reactant.@compile sync=true grad_quad_fun(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)
            grad_hexa_compiled = Reactant.@compile sync=true grad_hexa_fun(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)

            grad_mono_R = grad_mono_compiled(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)
            grad_quad_R = grad_quad_compiled(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)
            grad_hexa_R = grad_hexa_compiled(monoR, quadR, hexaR, k_inputR, k_outputR, q_par, q_perp, WR)

            Reactant.synchronize(grad_mono_R)
            Reactant.synchronize(grad_quad_R)
            Reactant.synchronize(grad_hexa_R)

            @test Array(grad_mono_R) ≈ grad_mono_ref atol=1e-7 rtol=1e-7
            @test Array(grad_quad_R) ≈ grad_quad_ref atol=1e-7 rtol=1e-7
            @test Array(grad_hexa_R) ≈ grad_hexa_ref atol=1e-7 rtol=1e-7
        end
end
