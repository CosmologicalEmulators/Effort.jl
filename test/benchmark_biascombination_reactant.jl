using BenchmarkTools
using Reactant
using Enzyme

bench_seconds = parse(Float64, get(ENV, "EFFORT_BENCH_SECONDS", "5.0"))
bench_samples = parse(Int, get(ENV, "EFFORT_BENCH_SAMPLES", "100"))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = bench_seconds
BenchmarkTools.DEFAULT_PARAMETERS.samples = bench_samples
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

function biascombo_original(biases)
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases
    Array([
        1, b1, b1^2, b2, b1 * b2, b2^2, bs, b1 * bs, b2 * bs, bs^2,
        b3, b1 * b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4,
    ])
end

function biascombo_rewrite(biases)
    b1 = biases[1:1]
    b2 = biases[2:2]
    bs = biases[3:3]
    b3 = biases[4:4]
    alpha0 = biases[5:5]
    alpha2 = biases[6:6]
    alpha4 = biases[7:7]
    alpha6 = biases[8:8]
    sn = biases[9:9]
    sn2 = biases[10:10]
    sn4 = biases[11:11]

    onev = b1 .* 0 .+ 1
    return vcat(
        onev,
        b1,
        b1 .* b1,
        b2,
        b1 .* b2,
        b2 .* b2,
        bs,
        b1 .* bs,
        b2 .* bs,
        bs .* bs,
        b3,
        b1 .* b3,
        alpha0,
        alpha2,
        alpha4,
        alpha6,
        sn,
        sn2,
        sn4,
    )
end

primal_original(biases) = Reactant.@allowscalar sum(biascombo_original(biases))
primal_rewrite(biases) = sum(biascombo_rewrite(biases))

grad_original(biases) = Enzyme.gradient(Reverse, primal_original, biases)[1]
grad_rewrite(biases) = Enzyme.gradient(Reverse, primal_rewrite, biases)[1]

biases = [1.5, 0.5, 0.1, 0.2, 0.01, 0.02, 0.03, 0.04, 1.0, 2.0, 3.0]

Reactant.set_default_backend("cpu")
biasesR = Reactant.to_rarray(biases)

# compile once
compiled_primal_original = Reactant.@compile sync=true primal_original(biasesR)
compiled_primal_rewrite = Reactant.@compile sync=true primal_rewrite(biasesR)
compiled_grad_original = Reactant.@compile sync=true grad_original(biasesR)
compiled_grad_rewrite = Reactant.@compile sync=true grad_rewrite(biasesR)

# warmups
Reactant.synchronize(compiled_primal_original(biasesR))
Reactant.synchronize(compiled_primal_rewrite(biasesR))
Reactant.synchronize(compiled_grad_original(biasesR))
Reactant.synchronize(compiled_grad_rewrite(biasesR))

trial_primal_original = @benchmark begin
    y = $compiled_primal_original($biasesR)
    Reactant.synchronize(y)
end

trial_primal_rewrite = @benchmark begin
    y = $compiled_primal_rewrite($biasesR)
    Reactant.synchronize(y)
end

trial_grad_original = @benchmark begin
    g = $compiled_grad_original($biasesR)
    Reactant.synchronize(g)
end

trial_grad_rewrite = @benchmark begin
    g = $compiled_grad_rewrite($biasesR)
    Reactant.synchronize(g)
end

median_ms(trial) = BenchmarkTools.median(trial).time / 1e6

po = median_ms(trial_primal_original)
pn = median_ms(trial_primal_rewrite)
go = median_ms(trial_grad_original)
gn = median_ms(trial_grad_rewrite)

println("\n=== Bias-combination Reactant micro-benchmark ===")
println("BenchmarkTools: seconds=$(bench_seconds), samples=$(bench_samples), evals=1")
println("Primal (original + allowscalar): $(round(po, digits=6)) ms")
println("Primal (rewrite no allowscalar): $(round(pn, digits=6)) ms")
println("Gradient Enzyme (original + allowscalar): $(round(go, digits=6)) ms")
println("Gradient Enzyme (rewrite no allowscalar): $(round(gn, digits=6)) ms")
println("Primal speedup rewrite/original: $(round(po / pn, digits=3))x")
println("Grad speedup rewrite/original:   $(round(go / gn, digits=3))x")

