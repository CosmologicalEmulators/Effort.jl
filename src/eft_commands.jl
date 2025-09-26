"""
    get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators)
Compute the Pℓ array given the cosmological parameters array `cosmology`,
the bias array `bs`, the growth factor `D` and an `AbstractEmulator`.
"""
function get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; noise_kwargs...)

    P11_comp_array = get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pct)
    noise = cosmoemu.NoiseModel(cosmoemu.p11.kgrid; noise_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, noise)
    biases = cosmoemu.BiasCombination(bs)

    return stacked_array * biases
end

function get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::PℓNoiseEmulator)

    P11_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.Pct)
    sn_comp_array = get_component(cosmology, D, cosmoemu.Noise)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, sn_comp_array)
    biases = cosmoemu.BiasCombination(bs)

    return stacked_array * biases
end

function get_Pℓ_jacobian(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; noise_kwargs...)

    P11_comp_array = get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pct)
    noise = cosmoemu.NoiseModel(cosmoemu.p11.kgrid; noise_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, noise)
    biases = cosmoemu.BiasCombination(bs)
    jacbiases = cosmoemu.JacobianBiasCombination(bs)

    return stacked_array * biases, stacked_array * jacbiases
end

function get_Pℓ_jacobian(cosmology::Array, D, bs::Array, cosmoemu::PℓNoiseEmulator)

    P11_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.Pct)
    sn_comp_array = get_component(cosmology, D, cosmoemu.Noise)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, sn_comp_array)
    biases = cosmoemu.BiasCombination(bs)
    jacbiases = cosmoemu.JacobianBiasCombination(bs)

    return stacked_array * biases, stacked_array * jacbiases
end

#function get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid::Array; k_nl=0.7)
#    P_stoch_0 = @. 1 / n_bar * (cϵ0 + cϵ1 * (k_grid / k_nl)^2)
#    P_stoch_2 = @. 1 / n_bar * (cϵ2 * (k_grid / k_nl)^2)
#    return P_stoch_0, P_stoch_2
#end

#function get_stoch_terms_jacobian(cϵ0, cϵ1, cϵ2, n_bar, k_grid::Array; k_nl=0.7)
#    myzeros = zeros(size(k_grid))
#    myones = ones(size(k_grid))
#    P_stoch_0 = @. 1 / n_bar * (cϵ0 + cϵ1 * (k_grid / k_nl)^2)
#    P_stoch_2 = @. 1 / n_bar * (cϵ2 * (k_grid / k_nl)^2)

#    ∂P0_∂cϵ0 = @. 1 / n_bar * myones
#    ∂P0_∂cϵ1 = @. 1 / n_bar * (k_grid / k_nl)^2
#    ∂P2_∂cϵ2 = @. 1 / n_bar * (k_grid / k_nl)^2

#    jac_P0 = hcat(∂P0_∂cϵ0, ∂P0_∂cϵ1, myzeros)
#    jac_P2 = hcat(myzeros, myzeros, ∂P2_∂cϵ2)
#    return P_stoch_0, P_stoch_2, jac_P0, jac_P2
#end
