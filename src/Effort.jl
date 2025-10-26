module Effort

using Base: @kwdef
# Load all dependencies needed for BackgroundCosmologyExt extension to activate
using DataInterpolations, FastGaussQuadrature, Integrals, LinearAlgebra, OrdinaryDiffEqTsit5, SciMLSensitivity
using AbstractCosmologicalEmulators
using AbstractCosmologicalEmulators: get_emulator_description
using Artifacts
using ChainRulesCore
using FindFirstFunctions
using LegendrePolynomials
using LoopVectorization
using Memoization
using NPZ
using SparseArrays
using Tullio
using Zygote
import JSON.parsefile
using Zygote: @adjoint

# Get the BackgroundCosmologyExt extension
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

# Import from extension if available
if !isnothing(ext)
    using .ext: AbstractCosmology, w0waCDMCosmology, D_z, D_f_z, f_z, E_z, d̃A_z
    # Re-export background cosmology functions for user convenience
    export AbstractCosmology, w0waCDMCosmology, D_z, D_f_z, f_z, E_z, d̃A_z
else
    @warn "BackgroundCosmologyExt extension not loaded. Background cosmology functions will not be available."
end

function __init__()
    global trained_emulators = Dict()
    trained_emulators["PyBirdmnuw0wacdm"] = Dict()
    trained_emulators["PyBirdmnuw0wacdm"]["0"] = load_multipole_emulator(joinpath(artifact"PyBirdmnuw0wacdm", "0/"))
    trained_emulators["PyBirdmnuw0wacdm"]["2"] = load_multipole_emulator(joinpath(artifact"PyBirdmnuw0wacdm", "2/"))
    trained_emulators["PyBirdmnuw0wacdm"]["4"] = load_multipole_emulator(joinpath(artifact"PyBirdmnuw0wacdm", "4/"))
end

include("neural_networks.jl")
include("eft_commands.jl")
include("projection.jl")
include("utils.jl")
include("chainrules.jl")

# Export main user-facing functions
export get_Pℓ, get_Pℓ_jacobian
export apply_AP, apply_AP_check, q_par_perp
export window_convolution
export PℓEmulator, ComponentEmulator
export load_component_emulator, load_multipole_emulator

end # module
