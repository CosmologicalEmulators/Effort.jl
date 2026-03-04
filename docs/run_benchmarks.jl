"""
Run and save benchmarks for Effort.jl documentation.

This script runs comprehensive performance benchmarks and saves them to disk
using BenchmarkTools' native save format. The saved benchmarks are then loaded
during documentation builds, avoiding the need to run expensive computations
during CI/CD.

Usage:
    julia --project=docs docs/run_benchmarks.jl

Output: docs/src/assets/effort_benchmark.json
"""

using BenchmarkTools
using Effort
using Printf
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote
using Mooncake
import ADTypes: AutoMooncake
using LinearAlgebra

# Set output file (same location as existing benchmark file)
output_file = joinpath(@__DIR__, "src", "assets", "effort_benchmark.json")

println("=" ^ 70)
println("Running Effort.jl Benchmarks")
println("=" ^ 70)
println("\nThis may take several minutes...")
println("Output file: $output_file\n")

#=============================================================================
Setup
=============================================================================#

println("[Setup] Loading emulators and defining parameters...")

# Load emulators
monopole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["0"]
quadrupole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["2"]
hexadecapole_emu = Effort.trained_emulators["PyBirdmnuw0wacdm"]["4"]
k_grid = vec(monopole_emu.P11.kgrid)

# Define cosmology
cosmology = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.044, nₛ = 0.9649, h = 0.6736,
    ωb = 0.02237, ωc = 0.12, mν = 0.06,
    w0 = -1.0, wa = 0.0, ωk = 0.0
)

cosmo_ref = Effort.w0waCDMCosmology(
    ln10Aₛ = 3.0, nₛ = 0.96, h = 0.70,
    ωb = 0.022, ωc = 0.115, mν = 0.06,
    w0 = -0.95, wa = 0.0, ωk = 0.0
)

z = 0.8
bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]

# Compute growth factors simultaneously
D, f = Effort.D_f_z(z, cosmology)
bias_params[8] = f

# Build emulator input
emulator_input = [
    z, cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h * 100,
    cosmology.ωb, cosmology.ωc, cosmology.mν, cosmology.w0, cosmology.wa
]

# Compute multipoles for AP benchmarks
P0 = Effort.get_Pℓ(emulator_input, D, bias_params, monopole_emu)
P2 = Effort.get_Pℓ(emulator_input, D, bias_params, quadrupole_emu)
P4 = Effort.get_Pℓ(emulator_input, D, bias_params, hexadecapole_emu)

# Compute AP parameters
q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

# Window Function Setup (User specified: 60x1200 matrix for nk_dense=400)
nk_dense_legacy = 400
nk_out_legacy = 20 # 20 per multipole = 60 total
k_dense_legacy = collect(range(minimum(k_grid), maximum(k_grid), length=nk_dense_legacy))

# Window matrices (20x400 for each multipole to match the 60x1200 structure)
W0_legacy = randn(nk_out_legacy, nk_dense_legacy)
W2_legacy = randn(nk_out_legacy, nk_dense_legacy)
W4_legacy = randn(nk_out_legacy, nk_dense_legacy)

# Unified Architecture Setup
nk_in_u = 70
nk_dense_u = 400
nk_out_u = 20 
K_u = 64
k_min_u, k_max_u = 1e-3, 0.5
k_in_u = collect(range(k_min_u, k_max_u, length=nk_in_u))
k_dense_u = collect(range(k_min_u, k_max_u, length=nk_dense_u))

# Precompute unified plan (shared for simplicity in single-point benchmarks)
unified_plan = Effort.prepare_ap_window_chebyshev(W0_legacy, W2_legacy, W4_legacy, k_dense_u, k_min_u, k_max_u, K_u)

# AD Backends
backends = [
    ("ForwardDiff", AutoForwardDiff()),
    ("Zygote", AutoZygote()),
    ("Mooncake", AutoMooncake(; config=nothing))
]

println("✓ Setup complete\n")

#=============================================================================
Benchmark Suite
=============================================================================#

# Create BenchmarkGroup (flat structure - Trials directly under "Effort")
suite = BenchmarkGroup()
suite["Effort"] = BenchmarkGroup()

println("[1/12] Benchmarking growth factors D(z) and f(z) together...")
suite["Effort"]["D_f_z"] = @benchmark Effort.D_f_z($z, $cosmology)
println("   Median time: $(median(suite["Effort"]["D_f_z"]).time / 1e3) μs")

println("\n[2/12] Benchmarking monopole emulation...")
suite["Effort"]["monopole"] = @benchmark Effort.get_Pℓ($emulator_input, $D, $bias_params, $monopole_emu)
println("   Median time: $(median(suite["Effort"]["monopole"]).time / 1e3) μs")

println("\n[3/12] Benchmarking quadrupole emulation...")
suite["Effort"]["quadrupole"] = @benchmark Effort.get_Pℓ($emulator_input, $D, $bias_params, $quadrupole_emu)
println("   Median time: $(median(suite["Effort"]["quadrupole"]).time / 1e3) μs")

println("\n[4/12] Benchmarking hexadecapole emulation...")
suite["Effort"]["hexadecapole"] = @benchmark Effort.get_Pℓ($emulator_input, $D, $bias_params, $hexadecapole_emu)
println("   Median time: $(median(suite["Effort"]["hexadecapole"]).time / 1e3) μs")

println("\n[5/12] Benchmarking AP corrections (Gauss-Lobatto)...")
suite["Effort"]["apply_AP"] = @benchmark Effort.apply_AP(
    $k_grid, $k_grid, $P0, $P2, $P4, $q_par, $q_perp, n_GL_points=8
)
println("   Median time: $(median(suite["Effort"]["apply_AP"]).time / 1e3) μs")

println("\n[6/12] Benchmarking complete pipeline (growth → emulator → AP)...")
# Define the complete pipeline function (including Window Convolution)
function compute_multipoles_benchmark(cosmology, z, bias_params, cosmo_ref,
                                      monopole_emu, quadrupole_emu, hexadecapole_emu,
                                      k_dense, w0, w2, w4)
    D, f = Effort.D_f_z(z, cosmology)
    bias_with_f = copy(bias_params)
    bias_with_f[8] = f

    emulator_input = [
        z, cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h * 100,
        cosmology.ωb, cosmology.ωc, cosmology.mν, cosmology.w0, cosmology.wa
    ]

    P0 = Effort.get_Pℓ(emulator_input, D, bias_with_f, monopole_emu)
    P2 = Effort.get_Pℓ(emulator_input, D, bias_with_f, quadrupole_emu)
    P4 = Effort.get_Pℓ(emulator_input, D, bias_with_f, hexadecapole_emu)

    k_grid_local = vec(monopole_emu.P11.kgrid)
    q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)

    # 1. Apply AP on a dense grid (e.g. 400 points)
    P0_d, P2_d, P4_d = Effort.apply_AP(k_grid_local, k_dense, P0, P2, P4, q_par, q_perp)

    # 2. Convolve with window function
    P0_c = Effort.window_convolution(w0, P0_d)
    P2_c = Effort.window_convolution(w2, P2_d)
    P4_c = Effort.window_convolution(w4, P4_d)

    return P0_c, P2_c, P4_c
end

suite["Effort"]["complete_pipeline"] = @benchmark compute_multipoles_benchmark(
    $cosmology, $z, $bias_params, $cosmo_ref,
    $monopole_emu, $quadrupole_emu, $hexadecapole_emu,
    $k_dense_legacy, $W0_legacy, $W2_legacy, $W4_legacy
)
println("   Median time: $(median(suite["Effort"]["complete_pipeline"]).time / 1e3) μs")

#=============================================================================
Automatic Differentiation Benchmarks
=============================================================================#

# Define a loss function over ALL FREE parameters (including Window Convolution)
function full_pipeline_loss(all_params)
    # Unpack: first 8 are cosmological, next 10 are bias parameters (excluding f)
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params[1], nₛ = all_params[2], h = all_params[3],
        ωb = all_params[4], ωc = all_params[5], mν = all_params[6],
        w0 = all_params[7], wa = all_params[8], ωk = 0.0
    )

    # Run complete pipeline: ODE solve → 3 emulators → AP corrections → Window
    D_local, f_local = Effort.D_f_z(z, cosmo_local)

    # Reconstruct full bias vector
    bias_local = [all_params[9:15]..., f_local, all_params[16:18]...]

    emulator_input_local = [
        z, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h * 100,
        cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa
    ]

    P0_local = Effort.get_Pℓ(emulator_input_local, D_local, bias_local, monopole_emu)
    P2_local = Effort.get_Pℓ(emulator_input_local, D_local, bias_local, quadrupole_emu)
    P4_local = Effort.get_Pℓ(emulator_input_local, D_local, bias_local, hexadecapole_emu)

    q_par_local, q_perp_local = Effort.q_par_perp(z, cosmo_local, cosmo_ref)

    # 1. Apply AP on a dense grid
    P0_AP, P2_AP, P4_AP = Effort.apply_AP(k_grid, k_dense_legacy, P0_local, P2_local, P4_local, q_par_local, q_perp_local)

    # 2. Apply Window convolution
    P0_conv = Effort.window_convolution(W0_legacy, P0_AP)
    P2_conv = Effort.window_convolution(W2_legacy, P2_AP)
    P4_conv = Effort.window_convolution(W4_legacy, P4_AP)

    # Return L2 norm of convolved multipoles
    return sum(abs2, P0_conv) + sum(abs2, P2_conv) + sum(abs2, P4_conv)
end

# Pack ALL FREE parameters into a single vector (8 cosmological + 10 bias = 18 total)
bias_params_no_f = [bias_params[1:7]..., bias_params[9:11]...]  # Exclude f at position 8
all_params = vcat(
    [cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h,
     cosmology.ωb, cosmology.ωc, cosmology.mν,
     cosmology.w0, cosmology.wa],
    bias_params_no_f
)

# Multi-z shared setup
z_array = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
all_params_multi = vcat(
    [cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h,
     cosmology.ωb, cosmology.ωc, cosmology.mν,
     cosmology.w0, cosmology.wa],
    repeat(bias_params_no_f, 6)
)

for (name, backend) in backends
    println("\n[7/12] Benchmarking $name gradient (w.r.t. all parameters)...")
    try
        # Prepare gradient (precomputes tape/cache)
        prep = DifferentiationInterface.prepare_gradient(full_pipeline_loss, backend, all_params)
        storage = similar(all_params)
        DifferentiationInterface.gradient!(full_pipeline_loss, storage, prep, backend, all_params)
        
        suite["Effort"]["gradient_$(lowercase(name))"] = @benchmark DifferentiationInterface.gradient!($full_pipeline_loss, $storage, $prep, $backend, $all_params)
        println("   Median time: $(median(suite["Effort"]["gradient_$(lowercase(name))"]).time / 1e3) μs")
    catch e
        println("   ⚠ $name failed: $(typeof(e))")
    end
end

#=============================================================================
Multi-Redshift Benchmarks
=============================================================================#

# Define Multi-z Forward Legacy (Iterative Dense AP + Window)
function multi_z_forward_legacy(all_params_multi)
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params_multi[1], nₛ = all_params_multi[2], h = all_params_multi[3],
        ωb = all_params_multi[4], ωc = all_params_multi[5], mν = all_params_multi[6],
        w0 = all_params_multi[7], wa = all_params_multi[8], ωk = 0.0
    )
    D_array, f_array = Effort.D_f_z(z_array, cosmo_local)
    total_loss = 0.0
    for (i, z_i) in enumerate(z_array)
        bias_start = 8 + (i-1)*10 + 1
        bias_end = 8 + i*10
        bias_this_z = [all_params_multi[bias_start:bias_start+6]..., f_array[i], all_params_multi[bias_start+7:bias_end]...]
        eval_in = [z_i, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h*100, cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa]
        P0_l = Effort.get_Pℓ(eval_in, D_array[i], bias_this_z, monopole_emu); P2_l = Effort.get_Pℓ(eval_in, D_array[i], bias_this_z, quadrupole_emu); P4_l = Effort.get_Pℓ(eval_in, D_array[i], bias_this_z, hexadecapole_emu)
        qpa, qpe = Effort.q_par_perp(z_i, cosmo_local, cosmo_ref)
        P0_A, P2_A, P4_A = Effort.apply_AP(k_grid, k_dense_legacy, P0_l, P2_l, P4_l, qpa, qpe)
        total_loss += sum(abs2, Effort.window_convolution(W0_legacy, P0_A)) + sum(abs2, Effort.window_convolution(W2_legacy, P2_A)) + sum(abs2, Effort.window_convolution(W4_legacy, P4_A))
    end
    return total_loss
end

# Define Multi-z Forward Unified (Chebyshev Optimised)
# We assume unified_plans is a vector of plans, one per redshift, or reuse a shared one for scaling tests
# For this benchmark, we'll reuse the unified_plan precomputed for scaling analysis
function multi_z_forward_unified(all_params_multi)
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params_multi[1], nₛ = all_params_multi[2], h = all_params_multi[3],
        ωb = all_params_multi[4], ωc = all_params_multi[5], mν = all_params_multi[6],
        w0 = all_params_multi[7], wa = all_params_multi[8], ωk = 0.0
    )
    D_array, f_array = Effort.D_f_z(z_array, cosmo_local)
    total_loss = 0.0
    for (i, z_i) in enumerate(z_array)
        bias_start = 8 + (i-1)*10 + 1
        bias_end = 8 + i*10
        bias_this_z = [all_params_multi[bias_start:bias_start+6]..., f_array[i], all_params_multi[bias_start+7:bias_end]...]
        eval_in = [z_i, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h*100, cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa]
        P0_l = Effort.get_Pℓ(eval_in, D_array[i], bias_this_z, monopole_emu); P2_l = Effort.get_Pℓ(eval_in, D_array[i], bias_this_z, quadrupole_emu); P4_l = Effort.get_Pℓ(eval_in, D_array[i], bias_this_z, hexadecapole_emu)
        qpa, qpe = Effort.q_par_perp(z_i, cosmo_local, cosmo_ref)
        y0, y2, y4 = Effort.apply_AP_and_window(unified_plan, k_grid, P0_l, P2_l, P4_l, qpa, qpe)
        total_loss += sum(abs2, y0) + sum(abs2, y2) + sum(abs2, y4)
    end
    return total_loss
end

println("\n[10/11] Benchmarking multi-redshift pipelines...")
suite["Effort"]["multi_z_legacy"] = @benchmark multi_z_forward_legacy($all_params_multi)
println("   Legacy Median time: $(median(suite["Effort"]["multi_z_legacy"]).time / 1e3) μs")
suite["Effort"]["multi_z_unified"] = @benchmark multi_z_forward_unified($all_params_multi)
println("   Unified Median time: $(median(suite["Effort"]["multi_z_unified"]).time / 1e3) μs")

for (name, backend) in backends
    println("\n[11/11] Benchmarking Multi-z $name gradient (Legacy)...")
    try
        prep_legacy = DifferentiationInterface.prepare_gradient(multi_z_forward_legacy, backend, all_params_multi)
        storage_multi = similar(all_params_multi)
        DifferentiationInterface.gradient!(multi_z_forward_legacy, storage_multi, prep_legacy, backend, all_params_multi)
        
        suite["Effort"]["multi_z_legacy_gradient_$(lowercase(name))"] = @benchmark DifferentiationInterface.gradient!($multi_z_forward_legacy, $storage_multi, $prep_legacy, $backend, $all_params_multi)
        println("   Legacy Gradient Median time: $(median(suite["Effort"]["multi_z_legacy_gradient_$(lowercase(name))"]).time / 1e3) μs")
    catch e
        println("   ⚠ $name failed (Legacy): $(typeof(e))")
    end

    println("\n[12/12] Benchmarking Multi-z $name gradient (Unified)...")
    try
        prep_unified = DifferentiationInterface.prepare_gradient(multi_z_forward_unified, backend, all_params_multi)
        storage_multi = similar(all_params_multi)
        DifferentiationInterface.gradient!(multi_z_forward_unified, storage_multi, prep_unified, backend, all_params_multi)
        
        suite["Effort"]["multi_z_unified_gradient_$(lowercase(name))"] = @benchmark DifferentiationInterface.gradient!($multi_z_forward_unified, $storage_multi, $prep_unified, $backend, $all_params_multi)
        println("   Unified Gradient Median time: $(median(suite["Effort"]["multi_z_unified_gradient_$(lowercase(name))"]).time / 1e3) μs")
    catch e
        println("   ⚠ $name failed (Unified): $(typeof(e))")
    end
end

#=============================================================================
Unified Pipeline Benchmarks
=============================================================================#

println("\n[13/15] Setting up unified AP + Window benchmarks...")

# Parameters from user request
nk_in_u = 70
nk_dense_u = 400
nk_out_u = 20 # 20 points * 3 multipoles = 60 output elements
K_u = 64
k_min_u, k_max_u = 1e-3, 0.5

k_in_u = collect(range(k_min_u, k_max_u, length=nk_in_u))
k_dense_u = collect(range(k_min_u, k_max_u, length=nk_dense_u))

# Mock data
m0_u = randn(nk_in_u)
m2_u = randn(nk_in_u)
m4_u = randn(nk_in_u)

# Window matrices (20x400 for each multipole)
W0_u = randn(nk_out_u, nk_dense_u)
W2_u = randn(nk_out_u, nk_dense_u)
W4_u = randn(nk_out_u, nk_dense_u)

# Precompute plan
unified_plan = Effort.prepare_ap_window_chebyshev(W0_u, W2_u, W4_u, k_dense_u, k_min_u, k_max_u, K_u)

println("\n[14/15] Benchmarking Unified AP + Window Pipeline...")
suite["Effort"]["unified_pipeline"] = @benchmark Effort.apply_AP_and_window(
    $unified_plan, $k_in_u, $m0_u, $m2_u, $m4_u, $q_par, $q_perp
)
println("   Median time: $(median(suite["Effort"]["unified_pipeline"]).time / 1e3) μs")

println("\n[15/15] Benchmarking Batched Unified Pipeline (n=10)...")
n_batch_u = 10
m0_batch_u = randn(nk_in_u, n_batch_u)
m2_batch_u = randn(nk_in_u, n_batch_u)
m4_batch_u = randn(nk_in_u, n_batch_u)

suite["Effort"]["unified_batched"] = @benchmark Effort.apply_AP_and_window(
    $unified_plan, $k_in_u, $m0_batch_u, $m2_batch_u, $m4_batch_u, $q_par, $q_perp
)
println("   Median time: $(median(suite["Effort"]["unified_batched"]).time / 1e3) μs")

#=============================================================================
Save Results
=============================================================================#

println("\n" * "=" ^ 70)
println("Saving benchmarks to disk...")

# Collect system metadata
using Dates
using JSON
metadata_dict = Dict(
    "julia_version" => string(VERSION),
    "effort_version" => string(Effort),
    "timestamp" => string(now()),
    "hostname" => gethostname(),
    "cpu_info" => Sys.cpu_info()[1].model,
    "ncores" => Sys.CPU_THREADS
)

# Save benchmarks using BenchmarkTools native format
BenchmarkTools.save(output_file, suite)

# Save metadata separately as JSON
metadata_file = joinpath(dirname(output_file), "benchmark_metadata.json")
open(metadata_file, "w") do io
    JSON.print(io, metadata_dict, 2)
end

println("✓ Benchmarks saved to: $output_file")
println("✓ Metadata saved to: $metadata_file")
println("\nSystem Information:")
println("  Julia version: $(metadata_dict["julia_version"])")
println("  CPU: $(metadata_dict["cpu_info"])")
println("  Cores: $(metadata_dict["ncores"])")

#=============================================================================
Summary
=============================================================================#

println("\n" * "=" ^ 70)
println("Benchmark Summary")
println("=" ^ 70)
println()

function format_time(ns)
    if ns < 1000
        return @sprintf("%.1f ns", ns)
    elseif ns < 1_000_000
        return @sprintf("%.2f μs", ns / 1000)
    elseif ns < 1_000_000_000
        return @sprintf("%.2f ms", ns / 1_000_000)
    else
        return @sprintf("%.2f s", ns / 1_000_000_000)
    end
end

function format_memory(bytes)
    if bytes < 1024
        return "$bytes bytes"
    elseif bytes < 1024^2
        return @sprintf("%.2f KB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.2f MB", bytes / 1024^2)
    else
        return @sprintf("%.2f GB", bytes / 1024^3)
    end
end

benchmark_specs = [
    ("D_f_z", "Growth factors D(z) and f(z)"),
    ("monopole", "Monopole (ℓ=0)"),
    ("quadrupole", "Quadrupole (ℓ=2)"),
    ("hexadecapole", "Hexadecapole (ℓ=4)"),
    ("apply_AP", "AP corrections (3 multipoles)"),
    ("complete_pipeline", "Complete pipeline (all steps)"),
    ("gradient_forwarddiff", "ForwardDiff (DI) End-to-End Gradient (18 params)"),
    ("gradient_zygote", "Zygote (DI) End-to-End Gradient (18 params)"),
    ("gradient_mooncake", "Mooncake (DI) End-to-End Gradient (18 params)"),
    ("unified_pipeline", "Unified AP + Window Pipeline (Single)"),
    ("unified_batched", "Unified AP + Window Pipeline (Batch n=10)")
]

# Multi-redshift benchmarks
multi_z_specs = [
    ("multi_z_legacy", "Multi-z Legacy Forward (6 DESI z)"),
    ("multi_z_unified", "Multi-z Unified Forward (6 DESI z)"),
    ("multi_z_legacy_gradient_forwarddiff", "Multi-z Legacy FD Grad (68 params)"),
    ("multi_z_legacy_gradient_zygote", "Multi-z Legacy Zygote Grad (68 params)"),
    ("multi_z_legacy_gradient_mooncake", "Multi-z Legacy Mooncake Grad (68 params)"),
    ("multi_z_unified_gradient_forwarddiff", "Multi-z Unified FD Grad (68 params)"),
    ("multi_z_unified_gradient_zygote", "Multi-z Unified Zygote Grad (68 params)"),
    ("multi_z_unified_gradient_mooncake", "Multi-z Unified Mooncake Grad (68 params)")
]

for (name, description) in benchmark_specs
    trial = suite["Effort"][name]
    println("$description:")
    println("  Time:   $(format_time(median(trial).time))")
    println("  Memory: $(format_memory(median(trial).memory))")
    println("  Allocs: $(median(trial).allocs)")
    println()
end

# Print multi-redshift benchmarks if they exist
println("Multi-Redshift Benchmarks (5 redshifts):")
println()
for (name, description) in multi_z_specs
    if haskey(suite["Effort"], name)
        trial = suite["Effort"][name]
        println("$description:")
        println("  Time:   $(format_time(median(trial).time))")
        println("  Memory: $(format_memory(median(trial).memory))")
        println("  Allocs: $(median(trial).allocs)")
        println()
    else
        println("$description: Not available (benchmark failed)")
        println()
    end
end

println("=" ^ 70)
println("Next steps:")
println("  1. Review the benchmark results above")
println("  2. The documentation will automatically load these benchmarks")
println("  3. Re-run this script if you update the code or hardware")
println("=" ^ 70)
