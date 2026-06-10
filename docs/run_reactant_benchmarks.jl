"""
Run and save docs-facing Reactant/Enzyme benchmarks for Effort.jl.

This script intentionally saves compact summary statistics instead of raw
BenchmarkTools trials. Reactant benchmarks are backend- and hardware-sensitive,
and the docs only need stable, readable median/p95/compile-latency tables.

Usage:
    julia --project=docs docs/run_reactant_benchmarks.jl

Optional controls:
    EFFORT_BENCH_SECONDS=2.0 EFFORT_BENCH_SAMPLES=50 julia --project=docs docs/run_reactant_benchmarks.jl

Output:
    docs/src/assets/reactant_benchmark_summary.json
"""

using BenchmarkTools
using Dates
using JSON
using Statistics
using InteractiveUtils
using Effort
using AbstractCosmologicalEmulators
using Reactant
using Enzyme
using DifferentiationInterface
using ADTypes: AutoMooncake
using Mooncake

const OUTPUT_FILE = joinpath(@__DIR__, "src", "assets", "reactant_benchmark_summary.json")
const BENCH_SECONDS = parse(Float64, get(ENV, "EFFORT_BENCH_SECONDS", "2.0"))
const BENCH_SAMPLES = parse(Int, get(ENV, "EFFORT_BENCH_SAMPLES", "50"))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = BENCH_SECONDS
BenchmarkTools.DEFAULT_PARAMETERS.samples = BENCH_SAMPLES
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

median_ms(trial) = BenchmarkTools.median(trial).time / 1e6
function p95_ms(trial)
    times = sort(collect(trial.times))
    times[max(1, Int(cld(95 * length(times), 100)))] / 1e6
end
memory_kb(trial) = BenchmarkTools.median(trial).memory / 1024
allocs(trial) = BenchmarkTools.median(trial).allocs

function summarize_trial(trial; label, category, backend, unit="ms", note="")
    Dict(
        "label" => label,
        "category" => category,
        "backend" => backend,
        "unit" => unit,
        "median_ms" => median_ms(trial),
        "p95_ms" => p95_ms(trial),
        "memory_kb" => memory_kb(trial),
        "allocs" => allocs(trial),
        "samples" => length(trial),
        "evals" => trial.params.evals,
        "note" => note,
    )
end

function sync_all(x)
    if x isa Tuple
        foreach(sync_all, x)
    else
        Reactant.synchronize(x)
    end
    return x
end

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

    return Effort.window_convolution(W, P0_AP),
           Effort.window_convolution(W, P2_AP),
           Effort.window_convolution(W, P4_AP)
end

function pipeline_loss(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
    c0, c2, c4 = pipeline_outputs(cosmology, bias, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W)
    return sum(c0) + sum(c2) + sum(c4)
end

function jacobian_projection_outputs(J0, J2, J4, k_input, k_output, q_par, q_perp, W)
    J0_AP, J2_AP, J4_AP = Effort.apply_AP(
        k_input,
        k_output,
        J0,
        J2,
        J4,
        q_par,
        q_perp;
        n_GL_points=8,
        method=Effort.Cubic(),
    )

    return W * J0_AP, W * J2_AP, W * J4_AP
end

function jacobian_projection_loss(J0, J2, J4, k_input, k_output, q_par, q_perp, W)
    WJ0, WJ2, WJ4 = jacobian_projection_outputs(J0, J2, J4, k_input, k_output, q_par, q_perp, W)
    return sum(WJ0) + sum(WJ2) + sum(WJ4)
end

grad_cosmo_react(c, b, D, emu0, emu2, emu4, k_input, k_output, q_par, q_perp, W) =
    Enzyme.gradient(
        Reverse,
        pipeline_loss,
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
    Enzyme.gradient(
        Reverse,
        pipeline_loss,
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

println("=" ^ 70)
println("Running Effort.jl Reactant/Enzyme docs benchmarks")
println("=" ^ 70)
println("BenchmarkTools: seconds=$(BENCH_SECONDS), samples=$(BENCH_SAMPLES), evals=1")
println("Output file: $(OUTPUT_FILE)")

Reactant.set_default_backend("cpu")

const EMU_KEY = "VelocileptorsREPTmnuw0wacdm"
emu0_host = Effort.trained_emulators[EMU_KEY]["0"]
emu2_host = Effort.trained_emulators[EMU_KEY]["2"]
emu4_host = Effort.trained_emulators[EMU_KEY]["4"]
emu0_dev = AbstractCosmologicalEmulators.to_reactant(emu0_host)
emu2_dev = AbstractCosmologicalEmulators.to_reactant(emu2_host)
emu4_dev = AbstractCosmologicalEmulators.to_reactant(emu4_host)

cosmology = [0.8, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
bias = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
D = 0.75
q_par = 1.02
q_perp = 0.98
k_input = vec(emu0_host.P11.kgrid)
k_output = copy(k_input)
n_window = 40
W = reshape(cos.(range(0.0, 3.0, length=n_window * length(k_output))), n_window, length(k_output))

loss_host(c, b) = pipeline_loss(c, b, D, emu0_host, emu2_host, emu4_host, k_input, k_output, q_par, q_perp, W)
loss_host_cosmo(c) = loss_host(c, bias)
loss_host_bias(b) = loss_host(cosmology, b)

println("\n[1/7] Host forward and Mooncake gradient baselines...")
mooncake_backend = AutoMooncake(; config=Mooncake.Config())
mooncake_prep_cosmo = DifferentiationInterface.prepare_gradient(loss_host_cosmo, mooncake_backend, cosmology)
mooncake_prep_bias = DifferentiationInterface.prepare_gradient(loss_host_bias, mooncake_backend, bias)
grad_buf_cosmo = similar(cosmology)
grad_buf_bias = similar(bias)

loss_host(cosmology, bias)
DifferentiationInterface.gradient!(loss_host_cosmo, grad_buf_cosmo, mooncake_prep_cosmo, mooncake_backend, cosmology)
DifferentiationInterface.gradient!(loss_host_bias, grad_buf_bias, mooncake_prep_bias, mooncake_backend, bias)

trial_host_forward = @benchmark loss_host($cosmology, $bias)
trial_host_grad_cosmo = @benchmark DifferentiationInterface.gradient!($loss_host_cosmo, $grad_buf_cosmo, $mooncake_prep_cosmo, $mooncake_backend, $cosmology)
trial_host_grad_bias = @benchmark DifferentiationInterface.gradient!($loss_host_bias, $grad_buf_bias, $mooncake_prep_bias, $mooncake_backend, $bias)

println("[2/7] Reactant input transfer...")
cosmologyR = Reactant.to_rarray(cosmology)
biasR = Reactant.to_rarray(bias)
k_inputR = Reactant.to_rarray(k_input)
k_outputR = Reactant.to_rarray(k_output)
WR = Reactant.to_rarray(W)

println("[3/7] Reactant forward compile + steady benchmark...")
compiled_forward = nothing
compile_forward_ms = 1000 * @elapsed begin
    global compiled_forward = Reactant.@compile sync=true pipeline_loss(
        cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR
    )
    sync_all(compiled_forward(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR))
end
trial_reactant_forward = @benchmark begin
    y = $compiled_forward($cosmologyR, $biasR, $D, $emu0_dev, $emu2_dev, $emu4_dev, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(y)
end

println("[4/7] Reactant+Enzyme cosmology-gradient compile + steady benchmark...")
compiled_grad_cosmo = nothing
compile_grad_cosmo_ms = 1000 * @elapsed begin
    global compiled_grad_cosmo = Reactant.@compile sync=true grad_cosmo_react(
        cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR
    )
    sync_all(compiled_grad_cosmo(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR))
end
trial_reactant_grad_cosmo = @benchmark begin
    g = $compiled_grad_cosmo($cosmologyR, $biasR, $D, $emu0_dev, $emu2_dev, $emu4_dev, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(g)
end

println("[5/7] Reactant+Enzyme bias-gradient compile + steady benchmark...")
compiled_grad_bias = nothing
compile_grad_bias_ms = 1000 * @elapsed begin
    global compiled_grad_bias = Reactant.@compile sync=true grad_bias_react(
        cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR
    )
    sync_all(compiled_grad_bias(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR))
end
trial_reactant_grad_bias = @benchmark begin
    g = $compiled_grad_bias($cosmologyR, $biasR, $D, $emu0_dev, $emu2_dev, $emu4_dev, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(g)
end

println("[6/7] Matrix apply_AP on analytical bias Jacobians...")
_, J0 = Effort.get_Pℓ_jacobian(cosmology, D, bias, emu0_host)
_, J2 = Effort.get_Pℓ_jacobian(cosmology, D, bias, emu2_host)
_, J4 = Effort.get_Pℓ_jacobian(cosmology, D, bias, emu4_host)
trial_host_jacobian_projection = @benchmark jacobian_projection_loss($J0, $J2, $J4, $k_input, $k_output, $q_par, $q_perp, $W)

J0R = Reactant.to_rarray(J0)
J2R = Reactant.to_rarray(J2)
J4R = Reactant.to_rarray(J4)
compiled_jacobian_projection = nothing
compile_jacobian_projection_ms = 1000 * @elapsed begin
    global compiled_jacobian_projection = Reactant.@compile sync=true jacobian_projection_loss(
        J0R, J2R, J4R, k_inputR, k_outputR, q_par, q_perp, WR
    )
    sync_all(compiled_jacobian_projection(J0R, J2R, J4R, k_inputR, k_outputR, q_par, q_perp, WR))
end
trial_reactant_jacobian_projection = @benchmark begin
    y = $compiled_jacobian_projection($J0R, $J2R, $J4R, $k_inputR, $k_outputR, $q_par, $q_perp, $WR)
    Reactant.synchronize(y)
end

println("[7/7] Validation and save...")
host_forward = loss_host(cosmology, bias)
reactant_forward = Reactant.to_number(compiled_forward(cosmologyR, biasR, D, emu0_dev, emu2_dev, emu4_dev, k_inputR, k_outputR, q_par, q_perp, WR))
host_jacobian = jacobian_projection_loss(J0, J2, J4, k_input, k_output, q_par, q_perp, W)
reactant_jacobian = Reactant.to_number(compiled_jacobian_projection(J0R, J2R, J4R, k_inputR, k_outputR, q_par, q_perp, WR))

benchmarks = Dict(
    "host_forward" => summarize_trial(trial_host_forward; label="Host forward: emulator → AP → window", category="forward", backend="Julia"),
    "host_grad_cosmo" => summarize_trial(trial_host_grad_cosmo; label="Host gradient wrt cosmology", category="gradient", backend="Mooncake"),
    "host_grad_bias" => summarize_trial(trial_host_grad_bias; label="Host gradient wrt bias", category="gradient", backend="Mooncake"),
    "reactant_forward" => merge(
        summarize_trial(trial_reactant_forward; label="Reactant forward: emulator → AP → window", category="forward", backend="Reactant CPU"),
        Dict("compile_first_ms" => compile_forward_ms),
    ),
    "reactant_grad_cosmo" => merge(
        summarize_trial(trial_reactant_grad_cosmo; label="Reactant+Enzyme gradient wrt cosmology", category="gradient", backend="Reactant CPU + Enzyme"),
        Dict("compile_first_ms" => compile_grad_cosmo_ms),
    ),
    "reactant_grad_bias" => merge(
        summarize_trial(trial_reactant_grad_bias; label="Reactant+Enzyme gradient wrt bias", category="gradient", backend="Reactant CPU + Enzyme"),
        Dict("compile_first_ms" => compile_grad_bias_ms),
    ),
    "host_jacobian_projection" => summarize_trial(trial_host_jacobian_projection; label="Host matrix AP on bias Jacobians", category="jacobian", backend="Julia"),
    "reactant_jacobian_projection" => merge(
        summarize_trial(trial_reactant_jacobian_projection; label="Reactant matrix AP on bias Jacobians", category="jacobian", backend="Reactant CPU"),
        Dict("compile_first_ms" => compile_jacobian_projection_ms),
    ),
)

comparisons = Dict(
    "reactant_forward_vs_host_forward" => benchmarks["reactant_forward"]["median_ms"] / benchmarks["host_forward"]["median_ms"],
    "reactant_grad_cosmo_vs_host_mooncake" => benchmarks["reactant_grad_cosmo"]["median_ms"] / benchmarks["host_grad_cosmo"]["median_ms"],
    "reactant_grad_bias_vs_host_mooncake" => benchmarks["reactant_grad_bias"]["median_ms"] / benchmarks["host_grad_bias"]["median_ms"],
    "reactant_jacobian_projection_vs_host" => benchmarks["reactant_jacobian_projection"]["median_ms"] / benchmarks["host_jacobian_projection"]["median_ms"],
)

metadata = Dict(
    "timestamp" => string(now()),
    "julia_version" => string(VERSION),
    "cpu_info" => Sys.cpu_info()[1].model,
    "nthreads" => Threads.nthreads(),
    "reactant_backend" => "cpu",
    "emulator" => EMU_KEY,
    "bench_seconds" => BENCH_SECONDS,
    "bench_samples" => BENCH_SAMPLES,
    "n_k" => length(k_input),
    "n_window" => n_window,
    "n_bias" => length(bias),
)

validation = Dict(
    "host_forward" => host_forward,
    "reactant_forward" => reactant_forward,
    "abs_forward_difference" => abs(host_forward - reactant_forward),
    "host_jacobian_projection" => host_jacobian,
    "reactant_jacobian_projection" => reactant_jacobian,
    "abs_jacobian_projection_difference" => abs(host_jacobian - reactant_jacobian),
)

payload = Dict(
    "metadata" => metadata,
    "benchmarks" => benchmarks,
    "comparisons" => comparisons,
    "validation" => validation,
)

mkpath(dirname(OUTPUT_FILE))
open(OUTPUT_FILE, "w") do io
    JSON.print(io, payload, 4)
end

println("\nBenchmark summary:")
for key in ("host_forward", "reactant_forward", "host_grad_cosmo", "reactant_grad_cosmo", "host_grad_bias", "reactant_grad_bias", "host_jacobian_projection", "reactant_jacobian_projection")
    b = benchmarks[key]
    compile_text = haskey(b, "compile_first_ms") ? ", compile+first=$(round(b["compile_first_ms"], digits=2)) ms" : ""
    println("  $(rpad(key, 30)) median=$(round(b["median_ms"], digits=4)) ms, p95=$(round(b["p95_ms"], digits=4)) ms$(compile_text)")
end

println("\nRelative factors (lower is better):")
for (key, value) in sort(collect(comparisons); by=first)
    println("  $(key): $(round(value, sigdigits=4))x")
end

println("\nValidation:")
println("  |host - Reactant| forward = $(validation["abs_forward_difference"])")
println("  |host - Reactant| matrix-Jacobian projection = $(validation["abs_jacobian_projection_difference"])")
println("\n✓ Saved Reactant/Enzyme benchmark summary to: $(OUTPUT_FILE)")
