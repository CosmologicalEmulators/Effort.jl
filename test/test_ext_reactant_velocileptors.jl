using Test
using Reactant
using Effort
using AbstractCosmologicalEmulators

function _velocileptors_reactant_pipeline(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
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

@testset "Effort ExtReactant: shipped Velocileptors emulators" begin
    ext_reactant = Base.get_extension(Effort, :ExtReactant)
    @test !isnothing(ext_reactant)
    Reactant.set_default_backend("cpu")

    cosmology = [0.8, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    bias = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
    D = 0.75
    q_par = 1.02
    q_perp = 0.98

    for emu_key in ("VelocileptorsREPTmnuw0wacdm", "VelocileptorsLPTmnuw0wacdm")
        @testset "$emu_key Reactant compile + forward parity" begin
            @test haskey(Effort.trained_emulators, emu_key)

            emu0_host = Effort.trained_emulators[emu_key]["0"]
            emu2_host = Effort.trained_emulators[emu_key]["2"]
            emu4_host = Effort.trained_emulators[emu_key]["4"]

            emu0_dev = AbstractCosmologicalEmulators.to_reactant(emu0_host)
            emu2_dev = AbstractCosmologicalEmulators.to_reactant(emu2_host)
            emu4_dev = AbstractCosmologicalEmulators.to_reactant(emu4_host)

            k_input = vec(emu0_host.P11.kgrid)
            k_output = copy(k_input)
            n_window = 24
            W = reshape(cos.(range(0.0, 2.0, length=n_window * length(k_output))), n_window, length(k_output))

            c0_ref, c2_ref, c4_ref = _velocileptors_reactant_pipeline(
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

            compiled_outputs = Reactant.@compile sync=true _velocileptors_reactant_pipeline(
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

            c0_R, c2_R, c4_R = compiled_outputs(
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

            Reactant.synchronize(c0_R)
            Reactant.synchronize(c2_R)
            Reactant.synchronize(c4_R)

            @test Array(c0_R) ≈ c0_ref atol=1e-7 rtol=1e-7
            @test Array(c2_R) ≈ c2_ref atol=1e-7 rtol=1e-7
            @test Array(c4_R) ≈ c4_ref atol=1e-7 rtol=1e-7
        end
    end
end
