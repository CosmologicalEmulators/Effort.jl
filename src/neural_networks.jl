abstract type AbstractComponentEmulators end

@kwdef mutable struct P11Emulator <: AbstractComponentEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64} = zeros(8,2)
    OutMinMax::Array{Float64} = zeros(2499,2)
end

@kwdef mutable struct PloopEmulator <: AbstractComponentEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64} = zeros(8,2)
    OutMinMax::Array{Float64} = zeros(2499,2)
end

@kwdef mutable struct PctEmulator <: AbstractComponentEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64} = zeros(8,2)
    OutMinMax::Array{Float64} = zeros(2499,2)
end

@kwdef mutable struct NoiseEmulator <: AbstractComponentEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64} = zeros(8,2)
    OutMinMax::Array{Float64} = zeros(2499,2)
end

function get_component(input_params, comp_emu::AbstractComponentEmulators)
    input = deepcopy(input_params)
    norm_input = maximin(input, comp_emu.InMinMax)
    norm_output = Array(run_emulator(norm_input, comp_emu.TrainedEmulator))
    output = inv_maximin(norm_output, comp_emu.OutMinMax)
    As = exp(input_params[1])*1e-10
    output .*= As
    return reshape(output, length(comp_emu.kgrid), :)
end

function get_component(input_params, comp_emu::PloopEmulator)
    input = deepcopy(input_params)
    norm_input = maximin(input, comp_emu.InMinMax)
    norm_output = Array(run_emulator(norm_input, comp_emu.TrainedEmulator))
    output = inv_maximin(norm_output, comp_emu.OutMinMax)
    As = exp(input_params[1])*1e-10
    output .*= As^2
    return reshape(output, length(comp_emu.kgrid), :)
end

function get_component(input_params, comp_emu::NoiseEmulator)
    input = deepcopy(input_params)
    norm_input = maximin(input, comp_emu.InMinMax)
    norm_output = Array(run_emulator(norm_input, comp_emu.TrainedEmulator))
    output = inv_maximin(norm_output, comp_emu.OutMinMax)
    return reshape(output, length(comp_emu.kgrid), :)
end

abstract type AbstractPℓEmulators end

@kwdef mutable struct PℓEmulator <: AbstractPℓEmulators
    P11::P11Emulator
    Ploop::PloopEmulator
    Pct::PctEmulator
end

@kwdef mutable struct PℓNoiseEmulator <: AbstractPℓEmulators
    Pℓ::PℓEmulator
    Noise::NoiseEmulator
end

abstract type AbstractBinEmulators end

@kwdef mutable struct BinEmulator <: AbstractBinEmulators
    MonoEmulator::AbstractPℓEmulators
    QuadEmulator::AbstractPℓEmulators
    HexaEmulator::AbstractPℓEmulators
end
