abstract type AbstractComponentEmulators end

@kwdef struct ComponentEmulator <: AbstractComponentEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64}
    OutMinMax::Array{Float64}
    Postprocessing::Function
end

function get_component(input_params, D, comp_emu::AbstractComponentEmulators)
    input = deepcopy(input_params)
    norm_input = maximin(input, comp_emu.InMinMax)
    norm_output = Array(run_emulator(norm_input, comp_emu.TrainedEmulator))
    output = inv_maximin(norm_output, comp_emu.OutMinMax)
    postprocessed_output = comp_emu.Postprocessing(input_params, output, D, comp_emu)
    return reshape(postprocessed_output, length(comp_emu.kgrid), :)
end

abstract type AbstractPℓEmulators end

@kwdef struct PℓEmulator <: AbstractPℓEmulators
    P11::ComponentEmulator
    Ploop::ComponentEmulator
    Pct::ComponentEmulator
    StochModel::Function
    BiasCombination::Function
    JacobianBiasCombination::Function
end
