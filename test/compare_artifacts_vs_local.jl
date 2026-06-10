using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Effort, Artifacts, Test, LinearAlgebra

# Mapping from family name to (local_folder, artifact_name, subpath_in_artifact)
configs = [
    ("PyBirdmnuw0wacdm", "trained_effort_pybird_mnuw0wacdm", "PyBirdmnuw0wacdm", ""),
    ("VelocileptorsREPTmnuw0wacdm", "trained_effort_velocileptors_rept_mnuw0wacdm", "trained_effort_velocileptors_rept_mnuw0wacdm", ""),
    ("VelocileptorsLPTmnuw0wacdm", "trained_effort_velocileptors_lpt_mnuw0wacdm", "trained_effort_velocileptors_lpt_mnuw0wacdm", "")
]

ells = ["0", "2", "4"]
input = [1.2, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
bias = collect(range(0.1, 1.1, length=11))
D = 0.8

println("=== Effort.jl: Artifact vs Local Rewritten Model Comparison ===")

for (fam_key, local_folder, art_name, subpath) in configs
    println("\nFamily: $fam_key")

    # Get artifact path from Artifacts.toml
    artifacts_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    art_path_base = artifact_path(artifact_hash(art_name, artifacts_toml))
    art_path = joinpath(art_path_base, subpath)
    loc_path = joinpath(@__DIR__, "..", local_folder)

    for ell in ells
        println("  Multipole ℓ=$ell:")
        # Load models
        old_emu = Effort.load_multipole_emulator(joinpath(art_path, ell, ""))
        new_emu = Effort.load_multipole_emulator(joinpath(loc_path, ell, ""))

        # 1. get_Pℓ comparison
        p_old = Base.invokelatest(Effort.get_Pℓ, input, D, bias, old_emu)
        p_new = Base.invokelatest(Effort.get_Pℓ, input, D, bias, new_emu)
        err_p = maximum(abs.(p_old .- p_new))

        # 2. analytical Jacobian comparison
        _, j_old = Base.invokelatest(Effort.get_Pℓ_jacobian, input, D, bias, old_emu)
        _, j_new = Base.invokelatest(Effort.get_Pℓ_jacobian, input, D, bias, new_emu)
        # Convert sparse to dense if needed for comparison
        err_j = maximum(abs.(Matrix(j_old) .- Matrix(j_new)))

        # 3. stochmodel comparison
        k = old_emu.P11.kgrid
        s_old = Base.invokelatest(old_emu.StochModel, k)
        s_new = Base.invokelatest(new_emu.StochModel, k)
        err_s = maximum(abs.(s_old .- s_new))

        println("    Pℓ max diff:    $err_p")
        println("    Jac max diff:   $err_j")
        println("    Stoch max diff: $err_s")

        @test err_p < 1e-10
        @test err_j < 1e-10
        @test err_s < 1e-10
    end
end

println("\nVerification complete: all rewritten models match original artifacts.")
