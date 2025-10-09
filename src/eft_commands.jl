"""
    get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators)
Compute the Pℓ array given the cosmological parameters array `cosmology`,
the bias array `bs`, the growth factor `D` and an `AbstractEmulator`.
"""
function get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; stoch_kwargs...)

    P11_comp_array = get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pct)
    stoch_comp_array = cosmoemu.StochModel(cosmoemu.P11.kgrid; stoch_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, stoch_comp_array)
    biases = cosmoemu.BiasCombination(bs)

    return stacked_array * biases
end

function get_Pℓ_jacobian(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; stoch_kwargs...)

    P11_comp_array = get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pct)
    stoch_comp_array = cosmoemu.StochModel(cosmoemu.P11.kgrid; stoch_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, stoch_comp_array)
    biases = cosmoemu.BiasCombination(bs)
    jacbiases = cosmoemu.JacobianBiasCombination(bs)

    return stacked_array * biases, stacked_array * jacbiases
end
