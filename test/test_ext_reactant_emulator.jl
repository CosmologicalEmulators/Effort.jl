using Test
using Enzyme
using ForwardDiff
using Reactant
using Effort
using AbstractCosmologicalEmulators

if !@isdefined(EMULATOR_TEST_COSMOLOGY)
    include("test_fixtures.jl")
end

@testset "Effort ExtReactant: trained emulator + AP + window" begin
    ext_reactant = Base.get_extension(Effort, :ExtReactant)
    @test !isnothing(ext_reactant)
    Reactant.set_default_backend("cpu")

        emu0_host = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
        emu2_host = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
        emu4_host = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]

        emu0_dev = AbstractCosmologicalEmulators.to_reactant(emu0_host)
        emu2_dev = AbstractCosmologicalEmulators.to_reactant(emu2_host)
        emu4_dev = AbstractCosmologicalEmulators.to_reactant(emu4_host)

        cosmology = copy(EMULATOR_TEST_COSMOLOGY)
        bias = copy(EMULATOR_TEST_BIAS)
        D = EMULATOR_TEST_D_GROWTH
        k_input = vec(emu0_host.P11.kgrid)
        k_output = copy(k_input)

        q_par = 1.02
        q_perp = 0.98

        n_window = 40
        W = reshape(cos.(range(0.0, 3.0, length=n_window * length(k_output))), n_window, length(k_output))

        function trained_pipeline_outputs(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
            P0 = Effort.get_Pℓ(cosmology, D, bias, emu0)
            P2 = Effort.get_Pℓ(cosmology, D, bias, emu2)
            P4 = Effort.get_Pℓ(cosmology, D, bias, emu4)

            P0_AP, P2_AP, P4_AP = Effort.apply_AP(
                k_input,
                k_output,
                P0,
                P2,
                P4,
                q_par,
                q_perp;
                n_GL_points=8,
                method=Effort.Cubic(),
            )

            return Effort.window_convolution(W, P0_AP),
                Effort.window_convolution(W, P2_AP),
                Effort.window_convolution(W, P4_AP)
        end

        function trained_pipeline_loss(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
            c0, c2, c4 = trained_pipeline_outputs(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
            return sum(c0) + sum(c2) + sum(c4)
        end

        @testset "Reactant compile + forward parity" begin
            c0_ref, c2_ref, c4_ref = trained_pipeline_outputs(
                cosmology,
                bias,
                D,
                emu0_host,
                emu2_host,
                emu4_host,
                k_input,
                k_output,
                q_par,
                q_perp,
                W,
            )

            cosmologyR = Reactant.to_rarray(cosmology)
            biasR = Reactant.to_rarray(bias)
            k_inputR = Reactant.to_rarray(k_input)
            k_outputR = Reactant.to_rarray(k_output)
            WR = Reactant.to_rarray(W)

            compiled_outputs = Reactant.@compile sync=true trained_pipeline_outputs(
                cosmologyR,
                biasR,
                D,
                emu0_dev,
                emu2_dev,
                emu4_dev,
                k_inputR,
                k_outputR,
                q_par,
                q_perp,
                WR,
            )

            c0_R, c2_R, c4_R = compiled_outputs(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)
            Reactant.synchronize(c0_R)
            Reactant.synchronize(c2_R)
            Reactant.synchronize(c4_R)

            @test Array(c0_R) ≈ c0_ref atol=1e-7 rtol=1e-7
            @test Array(c2_R) ≈ c2_ref atol=1e-7 rtol=1e-7
            @test Array(c4_R) ≈ c4_ref atol=1e-7 rtol=1e-7
        end

        @testset "Compiled Enzyme gradients wrt cosmology and bias" begin
            grad_cosmo_ref = ForwardDiff.gradient(c -> trained_pipeline_loss(c, bias, D, emu0_host, emu2_host, emu4_host, k_input, k_output, q_par, q_perp, W), cosmology)
            grad_bias_ref = ForwardDiff.gradient(b -> trained_pipeline_loss(cosmology, b, D, emu0_host, emu2_host, emu4_host, k_input, k_output, q_par, q_perp, W), bias)

            grad_cosmo_fun(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W) =
                Enzyme.gradient(Reverse, trained_pipeline_loss,
                    cosmology,
                    Const(bias),
                    Const(D),
                    Const(emu0),
                    Const(emu2),
                    Const(emu4),
                    Const(k_input),
                    Const(k_output),
                    Const(q_par),
                    Const(q_perp),
                    Const(W),
                )[1]

            grad_bias_fun(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W) =
                Enzyme.gradient(Reverse, trained_pipeline_loss,
                    Const(cosmology),
                    bias,
                    Const(D),
                    Const(emu0),
                    Const(emu2),
                    Const(emu4),
                    Const(k_input),
                    Const(k_output),
                    Const(q_par),
                    Const(q_perp),
                    Const(W),
                )[2]

            cosmologyR = Reactant.to_rarray(cosmology)
            biasR = Reactant.to_rarray(bias)
            k_inputR = Reactant.to_rarray(k_input)
            k_outputR = Reactant.to_rarray(k_output)
            WR = Reactant.to_rarray(W)

            grad_cosmo_compiled = Reactant.@compile sync=true grad_cosmo_fun(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)
            grad_bias_compiled = Reactant.@compile sync=true grad_bias_fun(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)

            grad_cosmo_R = grad_cosmo_compiled(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)
            grad_bias_R = grad_bias_compiled(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)

            Reactant.synchronize(grad_cosmo_R)
            Reactant.synchronize(grad_bias_R)

            @test Array(grad_cosmo_R) ≈ grad_cosmo_ref atol=1e-6 rtol=1e-6
            @test Array(grad_bias_R) ≈ grad_bias_ref atol=1e-6 rtol=1e-6
        end
end
