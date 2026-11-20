using BenchmarkTools
using Statistics
using DifferentiationInterface
using ADTypes: AutoMooncake
using Mooncake
using Enzyme
using Reactant
using Effort
using AbstractCosmologicalEmulators

if !@isdefined(EMULATOR_TEST_COSMOLOGY)
    include("test_fixtures.jl")
end

bench_seconds = parse(Float64, get(ENV, "EFFORT_BENCH_SECONDS", "2.0"))
bench_samples = parse(Int, get(ENV, "EFFORT_BENCH_SAMPLES", "50"))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = bench_seconds
BenchmarkTools.DEFAULT_PARAMETERS.samples = bench_samples
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

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

function pipeline_outputs(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
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

    c0 = Effort.window_convolution(W, P0_AP)
    c2 = Effort.window_convolution(W, P2_AP)
    c4 = Effort.window_convolution(W, P4_AP)
    return c0, c2, c4
end

function pipeline_loss(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
    c0, c2, c4 = pipeline_outputs(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
    return sum(c0) + sum(c2) + sum(c4)
end

loss_host(c, b) = pipeline_loss(c, b, D, emu0_host, emu2_host, emu4_host, k_input, k_output, q_par, q_perp, W)
loss_host_cosmo(c) = loss_host(c, bias)
loss_host_bias(b) = loss_host(cosmology, b)

mooncake_backend = AutoMooncake(; config=Mooncake.Config())

mooncake_prep_cosmo = DifferentiationInterface.prepare_gradient(loss_host_cosmo, mooncake_backend, cosmology)
mooncake_prep_bias = DifferentiationInterface.prepare_gradient(loss_host_bias, mooncake_backend, bias)
grad_buf_cosmo = similar(cosmology)
grad_buf_bias = similar(bias)

# Warmups (JIT + backend setup)
_ = loss_host(cosmology, bias)
_ = DifferentiationInterface.gradient!(loss_host_cosmo, grad_buf_cosmo, mooncake_prep_cosmo, mooncake_backend, cosmology)
_ = DifferentiationInterface.gradient!(loss_host_bias, grad_buf_bias, mooncake_prep_bias, mooncake_backend, bias)

trial_primal = @benchmark loss_host($cosmology, $bias)
trial_mooncake_cosmo = @benchmark DifferentiationInterface.gradient!($loss_host_cosmo, $grad_buf_cosmo, $mooncake_prep_cosmo, $mooncake_backend, $cosmology)
trial_mooncake_bias = @benchmark DifferentiationInterface.gradient!($loss_host_bias, $grad_buf_bias, $mooncake_prep_bias, $mooncake_backend, $bias)

cosmologyR = Reactant.to_rarray(cosmology)
biasR = Reactant.to_rarray(bias)
k_inputR = Reactant.to_rarray(k_input)
k_outputR = Reactant.to_rarray(k_output)
WR = Reactant.to_rarray(W)

compiled_primal = nothing
compile_plus_first_primal_s = @elapsed begin
    global compiled_primal = Reactant.@compile sync=true pipeline_loss(
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
    y = compiled_primal(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)
    Reactant.synchronize(y)
end

trial_reactant_primal = @benchmark begin
    y = $compiled_primal($cosmologyR, $biasR, $D, $emu0_dev, $emu2_dev, $emu4_dev, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(y)
end

grad_cosmo_react(c, b, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W) =
    Enzyme.gradient(Reverse, pipeline_loss,
        c,
        Const(b),
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

grad_bias_react(c, b, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W) =
    Enzyme.gradient(Reverse, pipeline_loss,
        Const(c),
        b,
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

compiled_grad_cosmo = nothing
compile_plus_first_grad_cosmo_s = @elapsed begin
    global compiled_grad_cosmo = Reactant.@compile sync=true grad_cosmo_react(
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
    g = compiled_grad_cosmo(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)
    Reactant.synchronize(g)
end

compiled_grad_bias = nothing
compile_plus_first_grad_bias_s = @elapsed begin
    global compiled_grad_bias = Reactant.@compile sync=true grad_bias_react(
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
    g = compiled_grad_bias(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR)
    Reactant.synchronize(g)
end

trial_reactant_grad_cosmo = @benchmark begin
    g = $compiled_grad_cosmo($cosmologyR, $biasR, $D, $emu0_dev, $emu2_dev, $emu4_dev, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(g)
end

trial_reactant_grad_bias = @benchmark begin
    g = $compiled_grad_bias($cosmologyR, $biasR, $D, $emu0_dev, $emu2_dev, $emu4_dev, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(g)
end

median_ms(trial) = BenchmarkTools.median(trial).time / 1e6
function p95_ms(trial)
    t = sort(collect(trial.times))
    idx = max(1, Int(cld(95 * length(t), 100)))
    return t[idx] / 1e6
end

primal_ms = median_ms(trial_primal)
mooncake_cosmo_ms = median_ms(trial_mooncake_cosmo)
mooncake_bias_ms = median_ms(trial_mooncake_bias)
reactant_primal_ms = median_ms(trial_reactant_primal)
reactant_grad_cosmo_ms = median_ms(trial_reactant_grad_cosmo)
reactant_grad_bias_ms = median_ms(trial_reactant_grad_bias)

primal_p95_ms = p95_ms(trial_primal)
mooncake_cosmo_p95_ms = p95_ms(trial_mooncake_cosmo)
mooncake_bias_p95_ms = p95_ms(trial_mooncake_bias)
reactant_primal_p95_ms = p95_ms(trial_reactant_primal)
reactant_grad_cosmo_p95_ms = p95_ms(trial_reactant_grad_cosmo)
reactant_grad_bias_p95_ms = p95_ms(trial_reactant_grad_bias)

println("\n=== Effort trained emulator -> AP -> window benchmark ===")
println("BenchmarkTools settings: seconds=$(bench_seconds), samples=$(bench_samples), evals=1")
println("CPU primal (plain Julia):                      $(round(primal_ms, digits=3)) ms")
println("  p95:                                         $(round(primal_p95_ms, digits=3)) ms")
println("CPU gradient Mooncake wrt cosmology (prepared + preallocated): $(round(mooncake_cosmo_ms, digits=3)) ms")
println("  p95:                                         $(round(mooncake_cosmo_p95_ms, digits=3)) ms")
println("CPU gradient Mooncake wrt bias (prepared + preallocated):      $(round(mooncake_bias_ms, digits=3)) ms")
println("  p95:                                         $(round(mooncake_bias_p95_ms, digits=3)) ms")
println("Reactant primal compile+first call:            $(round(1000 * compile_plus_first_primal_s, digits=3)) ms")
println("Reactant primal steady-state:                  $(round(reactant_primal_ms, digits=3)) ms")
println("  p95:                                         $(round(reactant_primal_p95_ms, digits=3)) ms")
println("Reactant+Enzyme grad(cosmology) compile+first: $(round(1000 * compile_plus_first_grad_cosmo_s, digits=3)) ms")
println("Reactant+Enzyme grad(cosmology) steady-state:  $(round(reactant_grad_cosmo_ms, digits=3)) ms")
println("  p95:                                         $(round(reactant_grad_cosmo_p95_ms, digits=3)) ms")
println("Reactant+Enzyme grad(bias) compile+first:      $(round(1000 * compile_plus_first_grad_bias_s, digits=3)) ms")
println("Reactant+Enzyme grad(bias) steady-state:       $(round(reactant_grad_bias_ms, digits=3)) ms")
println("  p95:                                         $(round(reactant_grad_bias_p95_ms, digits=3)) ms")

println("\n--- Relative factors (lower is better) ---")
println("Mooncake grad(cosmology) / primal:             $(round(mooncake_cosmo_ms / primal_ms, digits=2))x")
println("Mooncake grad(bias) / primal:                  $(round(mooncake_bias_ms / primal_ms, digits=2))x")
println("Reactant primal steady / primal:               $(round(reactant_primal_ms / primal_ms, digits=2))x")
println("Reactant+Enzyme grad(cosmology) steady / Mooncake(cosmology): $(round(reactant_grad_cosmo_ms / mooncake_cosmo_ms, sigdigits=3))x")
println("Reactant+Enzyme grad(bias) steady / Mooncake(bias):            $(round(reactant_grad_bias_ms / mooncake_bias_ms, sigdigits=3))x")
