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
# Define the complete pipeline function
function compute_multipoles_benchmark(cosmology, z, bias_params, cosmo_ref,
                                      monopole_emu, quadrupole_emu, hexadecapole_emu)
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

    if cosmology !== cosmo_ref
        q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmo_ref)
        k_grid_local = vec(monopole_emu.P11.kgrid)
        P0, P2, P4 = Effort.apply_AP(k_grid_local, k_grid_local, P0, P2, P4, q_par, q_perp)
    end

    return P0, P2, P4
end

suite["Effort"]["complete_pipeline"] = @benchmark compute_multipoles_benchmark(
    $cosmology, $z, $bias_params, $cosmo_ref,
    $monopole_emu, $quadrupole_emu, $hexadecapole_emu
)
println("   Median time: $(median(suite["Effort"]["complete_pipeline"]).time / 1e3) μs")

#=============================================================================
Automatic Differentiation Benchmarks
=============================================================================#

println("\n[7/12] Benchmarking ForwardDiff gradient (w.r.t. all parameters)...")
using ForwardDiff

# Define a loss function over ALL parameters (cosmological + bias)
function full_pipeline_loss(all_params)
    # Unpack: first 8 are cosmological, next 11 are bias parameters
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params[1], nₛ = all_params[2], h = all_params[3],
        ωb = all_params[4], ωc = all_params[5], mν = all_params[6],
        w0 = all_params[7], wa = all_params[8], ωk = 0.0
    )

    # Run complete pipeline
    D_local, f_local = Effort.D_f_z(z, cosmo_local)

    # Bias parameters (9-19) with f replaced at index 8
    bias_local = [all_params[9:16]..., f_local, all_params[17:19]...]

    emulator_input_local = [
        z, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h * 100,
        cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa
    ]

    P0_local = Effort.get_Pℓ(emulator_input_local, D_local, bias_local, monopole_emu)

    return sum(abs2, P0_local)  # L2 norm
end

# Pack ALL parameters into a single vector (8 cosmological + 11 bias = 19 total)
all_params = vcat(
    [cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h,
     cosmology.ωb, cosmology.ωc, cosmology.mν,
     cosmology.w0, cosmology.wa],
    bias_params
)

suite["Effort"]["forwarddiff_gradient"] = @benchmark ForwardDiff.gradient($full_pipeline_loss, $all_params)
println("   Median time: $(median(suite["Effort"]["forwarddiff_gradient"]).time / 1e3) μs")

println("\n[8/11] Benchmarking Zygote gradient (w.r.t. all parameters)...")
using Zygote

suite["Effort"]["zygote_gradient"] = @benchmark Zygote.gradient($full_pipeline_loss, $all_params)
println("   Median time: $(median(suite["Effort"]["zygote_gradient"]).time / 1e3) μs")

#=============================================================================
Multi-Redshift Benchmarks
=============================================================================#

println("\n[9/11] Setting up multi-redshift benchmarks (5 redshifts)...")

# Define 5 redshifts between 0.8 and 1.9
z_array = range(0.8, 1.9, length=5)
println("   Redshifts: $(collect(z_array))")

# Multi-redshift forward pass function
function multi_z_forward(all_params_multi)
    # Unpack: first 8 are cosmological, next 55 are bias (11 × 5 redshifts)
    cosmo_local = Effort.w0waCDMCosmology(
        ln10Aₛ = all_params_multi[1], nₛ = all_params_multi[2], h = all_params_multi[3],
        ωb = all_params_multi[4], ωc = all_params_multi[5], mν = all_params_multi[6],
        w0 = all_params_multi[7], wa = all_params_multi[8], ωk = 0.0
    )

    # Compute D and f for ALL redshifts at once (single ODE solve!)
    D_array, f_array = Effort.D_f_z(z_array, cosmo_local)

    # Compute power spectra for all redshifts
    total_loss = 0.0
    for (i, z_i) in enumerate(z_array)
        # Extract bias parameters for this redshift (11 params per redshift)
        bias_start = 8 + (i-1)*11 + 1
        bias_end = 8 + i*11
        bias_this_z = [all_params_multi[bias_start:bias_start+6]...,
                       f_array[i],
                       all_params_multi[bias_start+7:bias_end]...]

        emulator_input_local = [
            z_i, cosmo_local.ln10Aₛ, cosmo_local.nₛ, cosmo_local.h * 100,
            cosmo_local.ωb, cosmo_local.ωc, cosmo_local.mν, cosmo_local.w0, cosmo_local.wa
        ]

        P0_local = Effort.get_Pℓ(emulator_input_local, D_array[i], bias_this_z, monopole_emu)
        total_loss += sum(abs2, P0_local)
    end

    return total_loss
end

# Create parameter vector: 8 cosmo + 55 bias (11 × 5) = 63 total
all_params_multi = vcat(
    [cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h,
     cosmology.ωb, cosmology.ωc, cosmology.mν,
     cosmology.w0, cosmology.wa],
    repeat(bias_params, 5)  # Same bias params for each redshift
)

println("   Total parameters: $(length(all_params_multi)) (8 cosmo + 55 bias)")

println("\n[10/11] Benchmarking multi-redshift forward pass...")
suite["Effort"]["multi_z_forward"] = @benchmark multi_z_forward($all_params_multi)
println("   Median time: $(median(suite["Effort"]["multi_z_forward"]).time / 1e3) μs")

println("\n[11/11] Benchmarking multi-redshift ForwardDiff gradient...")
try
    suite["Effort"]["multi_z_forwarddiff"] = @benchmark ForwardDiff.gradient($multi_z_forward, $all_params_multi)
    println("   Median time: $(median(suite["Effort"]["multi_z_forwarddiff"]).time / 1e3) μs")
catch e
    println("   ⚠ ForwardDiff failed: $(typeof(e))")
    println("   Error: $e")
end

println("\n[12/12] Benchmarking multi-redshift Zygote gradient...")
try
    suite["Effort"]["multi_z_zygote"] = @benchmark Zygote.gradient($multi_z_forward, $all_params_multi)
    println("   Median time: $(median(suite["Effort"]["multi_z_zygote"]).time / 1e3) μs")
catch e
    println("   ⚠ Zygote failed: $(typeof(e))")
    println("   Error: $e")
end

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
    ("forwarddiff_gradient", "ForwardDiff gradient (19 params: 8 cosmo + 11 bias)"),
    ("zygote_gradient", "Zygote gradient (19 params: 8 cosmo + 11 bias)")
]

# Multi-redshift benchmarks (conditionally included if they exist)
multi_z_specs = [
    ("multi_z_forward", "Multi-z forward pass (5 redshifts, 63 params)"),
    ("multi_z_forwarddiff", "Multi-z ForwardDiff gradient (63 params)"),
    ("multi_z_zygote", "Multi-z Zygote gradient (63 params)")
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
