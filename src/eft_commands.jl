"""
    get_Pℓ(cosmology::Array, bs::Array, f, cosmoemu::AbstractPℓEmulators)
Compute the Pℓ array given the cosmological parameters array `cosmology`,
the bias array `bs`, the growth factor `f` and an `AbstractEmulator`.
"""
function get_Pℓ(cosmology::Array, bs::Array, f, cosmoemu::AbstractPℓEmulators)

    P11_comp_array = get_component(cosmology, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, cosmoemu.Pct)

    return sum_Pℓ_components(P11_comp_array, Ploop_comp_array, Pct_comp_array, bs, f)
end

function get_Pℓ(cosmology::Array, bs::Array, cosmoemu::PℓEmulatorVelocileptor)

    P11_comp_array = get_component(cosmology, cosmoemu)

    return sum_Pℓ_components(P11_comp_array, bs)
end

function get_Xiℓ(cosmology::Array, bs::Array, cosmoemu::XiℓEmulatorVelocileptor)

    P11_comp_array = get_component(cosmology, cosmoemu)

    return sum_Pℓ_components(P11_comp_array, bs)
end

function get_Pℓ(cosmology::Array, bs::Array, f, cosmoemu::AbstractBinEmulators)

    mono = get_Pℓ(cosmology, bs, f, cosmoemu.MonoEmulator)
    quad = get_Pℓ(cosmology, bs, f, cosmoemu.QuadEmulator)
    hexa = get_Pℓ(cosmology, bs, f, cosmoemu.HexaEmulator)

    return vcat(mono', quad', hexa')
end

function sum_Pℓ_components(P11_comp_array::AbstractArray{T}, Ploop_comp_array,
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

function sum_Pℓ_components(P11_comp_array::AbstractArray{T}, bs) where {T}

    P11_array = Array{T}(zeros(length(P11_comp_array[1,:])))

    bias_multiplication!(P11_array, bs, P11_comp_array)

    return P11_array
end

function bias_multiplication!(input_array, bias_array, Pk_input)
    @avx for b in eachindex(bias_array)
        for k in eachindex(input_array)
            input_array[k] += bias_array[b]*Pk_input[b,k]
        end
    end
end

function get_stoch(cϵ0, cϵ1, cϵ2, n_bar, k_grid::Array, k_nl=0.7)
    P_stoch = zeros(3, length(k_grid))
    P_stoch[1,:] = @. 1/n_bar*(cϵ0 + cϵ1*(k_grid/k_nl)^2)
    P_stoch[2,:] = @. 1/n_bar*(cϵ2 * (k_grid/k_nl)^2)
    return P_stoch
end

function get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid::Array, k_nl=0.7)
    P_stoch_0 = @. 1/n_bar*(cϵ0 + cϵ1*(k_grid/k_nl)^2)
    P_stoch_2 = @. 1/n_bar*(cϵ2 * (k_grid/k_nl)^2)
    return P_stoch_0, P_stoch_2
end

function get_stoch_terms_binned_efficient(cϵ0, cϵ1, cϵ2, n_bar, k_edges::Array; k_nl=0.7)
    n_bins = length(k_edges)-1
    P_stoch_0_c, _ = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_edges, k_nl)
    mytype = typeof(P_stoch_0_c[1])
    P_stoch_0 = zeros(mytype, n_bins)
    P_stoch_2 = zeros(mytype, n_bins)
    for i in 1:n_bins
        bin_vol = (k_edges[i+1]^3-k_edges[i]^3)/3
        P_stoch_0[i] = 1/n_bar*(cϵ0 + cϵ1*(k_edges[i+1]^5-k_edges[i]^5)/(5*bin_vol*k_nl^2))
        P_stoch_2[i] = 1/n_bar*(cϵ2*(k_edges[i+1]^5-k_edges[i]^5)/(5*bin_vol*k_nl^2))
    end
    return P_stoch_0, P_stoch_2
end

function get_stoch_terms_binned(cϵ0, cϵ1, cϵ2, n_bar, k_edges::Array; k_nl=0.7, nk = 30)
    n_bins = length(k_edges)-1
    P_stoch_0_c, _ = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_edges, k_nl)
    mytype = typeof(P_stoch_0_c[1])
    P_stoch_0 = zeros(mytype, n_bins)
    P_stoch_2 = zeros(mytype, n_bins)
    for i in 1:n_bins
        k_integral = Array(LinRange(k_edges[i], k_edges[i+1], nk))
        k_integral2 = k_integral.^2
        bin_vol = (k_edges[i+1]^3-k_edges[i]^3)/3
        Δk = (k_edges[i+1]-k_edges[i])/nk
        P_stoch_0_temp = @. 1/n_bar*(cϵ0 + cϵ1*(k_integral/k_nl)^2)
        P_stoch_2_temp = @. 1/n_bar*(cϵ2 * (k_integral/k_nl)^2)
        for j in 1:nk
            P_stoch_0[i] += P_stoch_0_temp[j] * k_integral2[j] * Δk / bin_vol
            P_stoch_2[i] += P_stoch_2_temp[j] * k_integral2[j] * Δk / bin_vol
        end
    end
    return P_stoch_0, P_stoch_2
end

function create_bin_edges(k_grid)
    Δk = round(k_grid[2]-k_grid[1]; digits = 2)
    k_b_edges = zeros(length(k_grid)+1)
    for i in 1:length(k_grid)
        k_b_edges[i+1] = k_grid[i]+Δk/2
    end
    k_b_edges[1] = k_grid[1]-Δk/2
    return k_b_edges
end
