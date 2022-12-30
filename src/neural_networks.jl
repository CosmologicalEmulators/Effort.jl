function _maximin_input!(x, in_MinMax)
    for i in eachindex(x)
        x[i] -= in_MinMax[i,1]
        x[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function _inv_maximin_output!(x, out_MinMax)
    for i in eachindex(x)
        x[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        x[i] += out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end
#TODO: right now we support only SimpleChains emulators, but in the near future we may
#support as well Flux and Lux. This is the reason behind the struct SimpleChainsEmulator

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
end

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

function get_component(input_params, comp_emu::AbstractComponentEmulators)
    input = deepcopy(input_params)
    _maximin_input!(input, comp_emu.InMinMax)
    output = Array(_run_emulator(input, comp_emu.TrainedEmulator))
    _inv_maximin_output!(output, comp_emu.OutMinMax)
    As = exp(input_params[1])*1e-10
    output .*= As
    return reshape(output, Int(length(output)/length(comp_emu.kgrid)), :)
end

function get_component(input_params, comp_emu::PloopEmulator)
    input = deepcopy(input_params)
    _maximin_input!(input, comp_emu.InMinMax)
    output = Array(_run_emulator(input, comp_emu.TrainedEmulator))
    _inv_maximin_output!(output, comp_emu.OutMinMax)
    As = exp(input_params[1])*1e-10
    output .*= As^2
    return reshape(output, Int(length(output)/length(comp_emu.kgrid)), :)
end

function _run_emulator(input, trained_emulator::SimpleChainsEmulator)
    return trained_emulator.Architecture(input, trained_emulator.Weights)
end

abstract type AbstractPℓEmulators end

@kwdef mutable struct PℓEmulator <: AbstractPℓEmulators
    P11::P11Emulator
    Ploop::PloopEmulator
    Pct::PctEmulator
end
