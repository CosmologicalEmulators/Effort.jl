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
