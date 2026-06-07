using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Effort, Reactant, Enzyme, BenchmarkTools, AbstractCosmologicalEmulators, Statistics

Reactant.set_default_backend("cpu")

families = [
    ("PyBird", "trained_effort_pybird_mnuw0wacdm"),
    ("LPT", "trained_effort_velocileptors_lpt_mnuw0wacdm"),
    ("REPT", "trained_effort_velocileptors_rept_mnuw0wacdm")
]

input = [1.2, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
bias = collect(range(0.1, 1.1, length=11))
D = 0.8

function pipeline_loss(c, b, D, e0, e2, e4, k_in, k_out, q_par, q_perp, W)
    p0 = Effort.get_Pℓ(c, D, b, e0)
    p2 = Effort.get_Pℓ(c, D, b, e2)
    p4 = Effort.get_Pℓ(c, D, b, e4)
    
    a0, a2, a4 = Effort.apply_AP(k_in, k_out, p0, p2, p4, q_par, q_perp; n_GL_points=8, method=Effort.Cubic())
    
    c0 = Effort.window_convolution(W, a0)
    c2 = Effort.window_convolution(W, a2)
    c4 = Effort.window_convolution(W, a4)
    
    return sum(c0) + sum(c2) + sum(c4)
end

# Gradients wrt cosmology and bias for the full pipeline
grad_cosmo(c, b, D, e0, e2, e4, k_in, k_out, q_par, q_perp, W) = 
    Enzyme.gradient(Reverse, pipeline_loss, c, Const(b), Const(D), Const(e0), Const(e2), Const(e4), Const(k_in), Const(k_out), Const(q_par), Const(q_perp), Const(W))[1]

grad_bias(c, b, D, e0, e2, e4, k_in, k_out, q_par, q_perp, W) = 
    Enzyme.gradient(Reverse, pipeline_loss, Const(c), b, Const(D), Const(e0), Const(e2), Const(e4), Const(k_in), Const(k_out), Const(q_par), Const(q_perp), Const(W))[2]

function run_benchmark_family(name, folder, input, bias, D)
    println("\nBenchmarking Family: $name")
    path = joinpath(@__DIR__, "..", folder)
    
    # Load emulators
    e0_host = Effort.load_multipole_emulator(joinpath(path, "0", ""))
    e2_host = Effort.load_multipole_emulator(joinpath(path, "2", ""))
    e4_host = Effort.load_multipole_emulator(joinpath(path, "4", ""))
    
    # Grid setup
    k_in = vec(e0_host.P11.kgrid)
    k_out = collect(range(0.01, 0.2, length=500))
    W = rand(50, 500)
    q_par, q_perp = 1.02, 0.98
    
    # Move to device
    e0 = AbstractCosmologicalEmulators.to_reactant(e0_host)
    e2 = AbstractCosmologicalEmulators.to_reactant(e2_host)
    e4 = AbstractCosmologicalEmulators.to_reactant(e4_host)
    
    inputR = Reactant.to_rarray(input)
    biasR = Reactant.to_rarray(bias)
    k_inR = Reactant.to_rarray(k_in)
    k_outR = Reactant.to_rarray(k_out)
    WR = Reactant.to_rarray(W)
    
    # Compile
    println("  Compiling Primal...")
    c_primal = Base.invokelatest(Reactant.compile, pipeline_loss, (inputR, biasR, D, e0, e2, e4, k_inR, k_outR, q_par, q_perp, WR))
    
    println("  Compiling Grad(Cosmo)...")
    c_gradc  = Base.invokelatest(Reactant.compile, grad_cosmo, (inputR, biasR, D, e0, e2, e4, k_inR, k_outR, q_par, q_perp, WR))
    
    println("  Compiling Grad(Bias)...")
    c_gradb  = Base.invokelatest(Reactant.compile, grad_bias, (inputR, biasR, D, e0, e2, e4, k_inR, k_outR, q_par, q_perp, WR))
    
    # Benchmark
    b_primal = @benchmark begin y = Base.invokelatest($c_primal, $inputR, $biasR, $D, $e0, $e2, $e4, $k_inR, $k_outR, $q_par, $q_perp, $WR); Reactant.synchronize(y); end
    b_gradc  = @benchmark begin g = Base.invokelatest($c_gradc, $inputR, $biasR, $D, $e0, $e2, $e4, $k_inR, $k_outR, $q_par, $q_perp, $WR); Reactant.synchronize(g); end
    b_gradb  = @benchmark begin g = Base.invokelatest($c_gradb, $inputR, $biasR, $D, $e0, $e2, $e4, $k_inR, $k_outR, $q_par, $q_perp, $WR); Reactant.synchronize(g); end
    
    println("    Primal:      $(round(median(b_primal).time/1e6, digits=4)) ms")
    println("    Grad(Cosmo): $(round(median(b_gradc).time/1e6, digits=4)) ms")
    println("    Grad(Bias):  $(round(median(b_gradb).time/1e6, digits=4)) ms")
end

println("=== Effort.jl: Local Model Reactant + Enzyme Benchmarks ===")

for (name, folder) in families
    # Call the runner with invokelatest to enter a new world age after closures might have been loaded
    Base.invokelatest(run_benchmark_family, name, folder, input, bias, D)
end
