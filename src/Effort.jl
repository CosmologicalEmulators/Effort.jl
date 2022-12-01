module Effort

using Base: @kwdef
using LoopVectorization
using SimpleChains

function maximin_input!(x, in_MinMax)
    for i in eachindex(x)
        x[i] -= in_MinMax[i,1]
        x[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(x, out_MinMax)
    for i in eachindex(x)
        x[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        x[i] += out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end

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

function compute_component(input_params, comp_emu::AbstractComponentEmulators)
    input = deepcopy(input_params)
    maximin_input!(input, comp_emu.InMinMax)
    output = Array(run_emulator(input, comp_emu.TrainedEmulator))
    inv_maximin_output!(output, comp_emu.OutMinMax)
    As = exp(input_params[1])*1e-10
    output .*= As
    return reshape(output, Int(length(output)/length(comp_emu.kgrid)), :)
end

function compute_component(input_params, comp_emu::PloopEmulator)
    input = deepcopy(input_params)
    maximin_input!(input, comp_emu.InMinMax)
    output = Array(run_emulator(input, comp_emu.TrainedEmulator))
    inv_maximin_output!(output, comp_emu.OutMinMax)
    As = exp(input_params[1])*1e-10
    output .*= As^2
    return reshape(output, Int(length(output)/length(comp_emu.kgrid)), :)
end

function run_emulator(input, trained_emulator::SimpleChainsEmulator)
    return trained_emulator.Architecture(input, trained_emulator.Weights)
end

abstract type AbstractPℓEmulators end

@kwdef mutable struct PℓEmulator <: AbstractPℓEmulators
    P11::P11Emulator
    Ploop::PloopEmulator
    Pct::PctEmulator
end

function ComputePℓ(cosmology, bs, f, cosmoemu::AbstractPℓEmulators)

    P11_comp_array = compute_component(cosmology, cosmoemu.P11)
    Ploop_comp_array = compute_component(cosmology, cosmoemu.Ploop)
    Pct_comp_array = compute_component(cosmology, cosmoemu.Pct)

    return SumMultipoleComponents(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
end

function SumMultipoleComponents(P11_comp_array::AbstractArray{T}, Ploop_comp_array,
    Pct_comp_array, bs, f) where {T}
    b1, b2, b3, b4, b5, b6, b7 = bs

    b11 = Array([ b1^2, 2*b1*f, f^2])
    bloop = Array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
    bct = Array([ 2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7 ])

    P11_array = Array{T}(zeros(length(P11_comp_array[1,:])))
    Ploop_array = Array{T}(zeros(length(P11_comp_array[1,:])))
    Pct_array = Array{T}(zeros(length(P11_comp_array[1,:])))

    bias_multiplication!(P11_array, b11, P11_comp_array)
    bias_multiplication!(Ploop_array, bloop, Ploop_comp_array)
    bias_multiplication!(Pct_array, bct, Pct_comp_array)
    Pℓ = P11_array .+ Ploop_array .+ Pct_array

    return Pℓ
end

function bias_multiplication!(input_array, bias_array, Pk_input)
    @avx for b in eachindex(bias_array)
        for k in eachindex(input_array)
            input_array[k] += bias_array[b]*Pk_input[b,k]
        end
    end
end

end # module
