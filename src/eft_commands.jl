"""
    get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators)
Compute the Pℓ array given the cosmological parameters array `cosmology`,
the bias array `bs`, the growth factor `D` and an `AbstractEmulator`.
"""
function get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators)

    P11_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pℓ.Pct)
    sn_comp_array = get_component(cosmology, D, cosmoemu.Noise)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, sn_comp_array)
    biases = cosmoemu.BiasCombination(bs)

    return stacked_array * biases
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

function get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid::Array; k_nl=0.7)
    P_stoch_0 = @. 1 / n_bar * (cϵ0 + cϵ1 * (k_grid / k_nl)^2)
    P_stoch_2 = @. 1 / n_bar * (cϵ2 * (k_grid / k_nl)^2)
    return P_stoch_0, P_stoch_2
end

function get_stoch_terms_jacobian(cϵ0, cϵ1, cϵ2, n_bar, k_grid::Array; k_nl=0.7)
    myzeros = k_grid .* 0.0
    P_stoch_0 = @. 1 / n_bar * (cϵ0 + cϵ1 * (k_grid / k_nl)^2)
    P_stoch_2 = @. 1 / n_bar * (cϵ2 * (k_grid / k_nl)^2)

    ∂P0_∂cϵ0 = @. 1 / n_bar
    ∂P0_∂cϵ1 = @. 1 / n_bar * (k_grid / k_nl)^2
    ∂P2_∂cϵ2 = @. 1 / n_bar * (k_grid / k_nl)^2
    jac_P0 = hcat(∂P0_∂cϵ0, ∂P0_∂cϵ1, myzeros)
    jac_P2 = hcat(myzeros, myzeros, ∂P2_∂cϵ2)
    return P_stoch_0, P_stoch_2, jac_P0, jac_P2
end

function get_stoch_terms_binned_efficient(cϵ0, cϵ1, cϵ2, n_bar, k_edges::Array; k_nl=0.7)
    n_bins = length(k_edges) - 1
    P_stoch_0_c, _ = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_edges; k_nl=k_nl)
    mytype = typeof(P_stoch_0_c[1])
    P_stoch_0 = zeros(mytype, n_bins)
    P_stoch_2 = zeros(mytype, n_bins)
    for i in 1:n_bins
        bin_vol = (k_edges[i+1]^3 - k_edges[i]^3) / 3
        P_stoch_0[i] = 1 / n_bar * (cϵ0 + cϵ1 * (k_edges[i+1]^5 - k_edges[i]^5) / (5 * bin_vol * k_nl^2))
        P_stoch_2[i] = 1 / n_bar * (cϵ2 * (k_edges[i+1]^5 - k_edges[i]^5) / (5 * bin_vol * k_nl^2))
    end
    return P_stoch_0, P_stoch_2
end

function get_stoch_terms_binned(cϵ0, cϵ1, cϵ2, n_bar, k_edges::Array; k_nl=0.7, nk=30)
    n_bins = length(k_edges) - 1
    P_stoch_0_c, _ = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_edges; k_nl=k_nl)
    mytype = typeof(P_stoch_0_c[1])
    P_stoch_0 = zeros(mytype, n_bins)
    P_stoch_2 = zeros(mytype, n_bins)
    for i in 1:n_bins
        k_integral = Array(LinRange(k_edges[i], k_edges[i+1], nk))
        k_integral2 = k_integral .^ 2
        bin_vol = (k_edges[i+1]^3 - k_edges[i]^3) / 3
        Δk = (k_edges[i+1] - k_edges[i]) / nk
        P_stoch_0_temp = @. 1 / n_bar * (cϵ0 + cϵ1 * (k_integral / k_nl)^2)
        P_stoch_2_temp = @. 1 / n_bar * (cϵ2 * (k_integral / k_nl)^2)
        for j in 1:nk
            P_stoch_0[i] += P_stoch_0_temp[j] * k_integral2[j] * Δk / bin_vol
            P_stoch_2[i] += P_stoch_2_temp[j] * k_integral2[j] * Δk / bin_vol
        end
    end
    return P_stoch_0, P_stoch_2
end
