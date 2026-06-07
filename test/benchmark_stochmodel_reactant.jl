using BenchmarkTools
using Reactant
using Enzyme

bench_seconds = parse(Float64, get(ENV, "EFFORT_BENCH_SECONDS", "10.0"))
bench_samples = parse(Int, get(ENV, "EFFORT_BENCH_SAMPLES", "200"))

BenchmarkTools.DEFAULT_PARAMETERS.seconds = bench_seconds
BenchmarkTools.DEFAULT_PARAMETERS.samples = bench_samples
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

# -------------------------
# Original stoch-model forms
# -------------------------
function stoch0_original(k)
    comp0 = ones(length(k))
    comp2 = k .^ 2 ./ 3
    comp4 = k .^ 4 ./ 5
    return hcat(comp0, comp2, comp4)
end

function stoch2_original(k)
    comp0 = zeros(length(k))
    comp2 = 2 .* k .^ 2 ./ 3
    comp4 = 4 .* k .^ 4 ./ 7
    return hcat(comp0, comp2, comp4)
end

function stoch4_original(k)
    comp0 = zeros(length(k))
    comp2 = zeros(length(k))
    comp4 = 8 .* k .^ 4 ./ 35
    return hcat(comp0, comp2, comp4)
end

# -------------------------
# Traced-safe rewrites (no allowscalar needed)
# -------------------------
function stoch0_rewrite(k)
    zero = k .* 0
    one = zero .+ 1
    k2 = k .* k
    k4 = k2 .* k2
    return hcat(one, k2 ./ 3, k4 ./ 5)
end

function stoch2_rewrite(k)
    zero = k .* 0
    k2 = k .* k
    k4 = k2 .* k2
    return hcat(zero, 2 .* k2 ./ 3, 4 .* k4 ./ 7)
end

function stoch4_rewrite(k)
    zero = k .* 0
    k2 = k .* k
    k4 = k2 .* k2
    return hcat(zero, zero, 8 .* k4 ./ 35)
end

# Loss wrappers (scalar objective for primal/grad benchmarks)
primal0_original(k) = Reactant.@allowscalar sum(stoch0_original(k))
primal0_rewrite(k) = sum(stoch0_rewrite(k))
primal2_original(k) = Reactant.@allowscalar sum(stoch2_original(k))
primal2_rewrite(k) = sum(stoch2_rewrite(k))
primal4_original(k) = Reactant.@allowscalar sum(stoch4_original(k))
primal4_rewrite(k) = sum(stoch4_rewrite(k))

grad0_original(k) = Enzyme.gradient(Reverse, primal0_original, k)[1]
grad0_rewrite(k) = Enzyme.gradient(Reverse, primal0_rewrite, k)[1]
grad2_original(k) = Enzyme.gradient(Reverse, primal2_original, k)[1]
grad2_rewrite(k) = Enzyme.gradient(Reverse, primal2_rewrite, k)[1]
grad4_original(k) = Enzyme.gradient(Reverse, primal4_original, k)[1]
grad4_rewrite(k) = Enzyme.gradient(Reverse, primal4_rewrite, k)[1]

median_ms(trial) = BenchmarkTools.median(trial).time / 1e6

function run_case(label, primal_orig, primal_new, grad_orig, grad_new, kR)
    cpo = Reactant.@compile sync=true primal_orig(kR)
    cpn = Reactant.@compile sync=true primal_new(kR)
    cgo = Reactant.@compile sync=true grad_orig(kR)
    cgn = Reactant.@compile sync=true grad_new(kR)

    # warmup
    Reactant.synchronize(cpo(kR))
    Reactant.synchronize(cpn(kR))
    Reactant.synchronize(cgo(kR))
    Reactant.synchronize(cgn(kR))

    tpo = @benchmark begin
        y = $cpo($kR)
        Reactant.synchronize(y)
    end

    tpn = @benchmark begin
        y = $cpn($kR)
        Reactant.synchronize(y)
    end

    tgo = @benchmark begin
        g = $cgo($kR)
        Reactant.synchronize(g)
    end

    tgn = @benchmark begin
        g = $cgn($kR)
        Reactant.synchronize(g)
    end

    go_val = Array(cgo(kR))
    gn_val = Array(cgn(kR))
    max_grad_diff = maximum(abs.(go_val .- gn_val))

    po = median_ms(tpo)
    pn = median_ms(tpn)
    go = median_ms(tgo)
    gn = median_ms(tgn)

    println("\n-- ℓ = $(label) --")
    println("Primal  original(+allowscalar): $(round(po, digits=6)) ms")
    println("Primal  rewrite (no allowscalar): $(round(pn, digits=6)) ms")
    println("Grad    original(+allowscalar): $(round(go, digits=6)) ms")
    println("Grad    rewrite (no allowscalar): $(round(gn, digits=6)) ms")
    println("Primal speedup rewrite/original: $(round(po / pn, digits=3))x")
    println("Grad   speedup rewrite/original: $(round(go / gn, digits=3))x")
    println("max |Δ grad| original vs rewrite: $(max_grad_diff)")
end

Reactant.set_default_backend("cpu")
k = collect(range(0.01, 0.4, length=256))
kR = Reactant.to_rarray(k)

println("\n=== Stoch-model Reactant micro-benchmark (compiled) ===")
println("BenchmarkTools: seconds=$(bench_seconds), samples=$(bench_samples), evals=1")

run_case("0", primal0_original, primal0_rewrite, grad0_original, grad0_rewrite, kR)
run_case("2", primal2_original, primal2_rewrite, grad2_original, grad2_rewrite, kR)
run_case("4", primal4_original, primal4_rewrite, grad4_original, grad4_rewrite, kR)

