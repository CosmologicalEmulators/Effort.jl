using BenchmarkTools
using Reactant
using Enzyme

bench_seconds = parse(Float64, get(ENV, "EFFORT_BENCH_SECONDS", "10.0"))
bench_samples = parse(Int, get(ENV, "EFFORT_BENCH_SAMPLES", "200"))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = bench_seconds
BenchmarkTools.DEFAULT_PARAMETERS.samples = bench_samples
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

# Original closure style (needs allowscalar because of input[2]).
postproc_original(input, output, D) = output .* (exp(input[2]) * 1e-10 .* D^2) .^ 2

# Rewrite: traced-safe indexing and broadcast-only math.
function postproc_rewrite(input, output, D)
    ln10As = input[2:2]
    amp = exp.(ln10As) .* 1e-10 .* (D * D)
    return output .* (amp .* amp)
end

primal_original(input, output, D) = Reactant.@allowscalar sum(postproc_original(input, output, D))
primal_rewrite(input, output, D) = sum(postproc_rewrite(input, output, D))

grad_original(input, output, D) = Enzyme.gradient(Reverse, primal_original, input, Const(output), Const(D))[1]
grad_rewrite(input, output, D) = Enzyme.gradient(Reverse, primal_rewrite, input, Const(output), Const(D))[1]

input = [1.2, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
output = collect(range(1e-2, 1.0, length=40))
D = 0.8

Reactant.set_default_backend("cpu")
inputR = Reactant.to_rarray(input)
outputR = Reactant.to_rarray(output)

# compile once
compiled_primal_original = Reactant.@compile sync=true primal_original(inputR, outputR, D)
compiled_primal_rewrite = Reactant.@compile sync=true primal_rewrite(inputR, outputR, D)
compiled_grad_original = Reactant.@compile sync=true grad_original(inputR, outputR, D)
compiled_grad_rewrite = Reactant.@compile sync=true grad_rewrite(inputR, outputR, D)

# warmups
Reactant.synchronize(compiled_primal_original(inputR, outputR, D))
Reactant.synchronize(compiled_primal_rewrite(inputR, outputR, D))
Reactant.synchronize(compiled_grad_original(inputR, outputR, D))
Reactant.synchronize(compiled_grad_rewrite(inputR, outputR, D))

trial_primal_original = @benchmark begin
    y = $compiled_primal_original($inputR, $outputR, $D)
    Reactant.synchronize(y)
end

trial_primal_rewrite = @benchmark begin
    y = $compiled_primal_rewrite($inputR, $outputR, $D)
    Reactant.synchronize(y)
end

trial_grad_original = @benchmark begin
    g = $compiled_grad_original($inputR, $outputR, $D)
    Reactant.synchronize(g)
end

trial_grad_rewrite = @benchmark begin
    g = $compiled_grad_rewrite($inputR, $outputR, $D)
    Reactant.synchronize(g)
end

median_ms(trial) = BenchmarkTools.median(trial).time / 1e6

po = median_ms(trial_primal_original)
pn = median_ms(trial_primal_rewrite)
go = median_ms(trial_grad_original)
gn = median_ms(trial_grad_rewrite)

# quick consistency check for gradient
go_val = Array(compiled_grad_original(inputR, outputR, D))
gn_val = Array(compiled_grad_rewrite(inputR, outputR, D))

println("\n=== Postprocessing Reactant micro-benchmark ===")
println("BenchmarkTools: seconds=$(bench_seconds), samples=$(bench_samples), evals=1")
println("Primal (original + allowscalar): $(round(po, digits=6)) ms")
println("Primal (rewrite no allowscalar): $(round(pn, digits=6)) ms")
println("Gradient Enzyme wrt input (original + allowscalar): $(round(go, digits=6)) ms")
println("Gradient Enzyme wrt input (rewrite no allowscalar): $(round(gn, digits=6)) ms")
println("Primal speedup rewrite/original: $(round(po / pn, digits=3))x")
println("Grad speedup rewrite/original:   $(round(go / gn, digits=3))x")
println("max |Δ grad|: $(maximum(abs.(go_val .- gn_val)))")
