function _Pkμ(k, μ, Int_Mono, Int_Quad, Int_Hexa)
    return Int_Mono(k)*_legendre_0(μ) + Int_Quad(k)*_legendre_2(μ) + Int_Hexa(k)*_legendre_4(μ)
end

function _k_true(k_o, μ_o, q_perp, F)
    return @. k_o/q_perp*sqrt(1+μ_o^2*(1/F^2-1))
end

function _μ_true(μ_o, F)
    return μ_o/F/sqrt(1+μ_o^2*(1/F^2-1))
end

function k_true(k_o::Array, μ_o::Array, q_perp, F)
    a = @. sqrt(1+μ_o^2*(1/F^2-1))
    result = (k_o./q_perp) * a'
    return vec(result)
end

function μ_true(μ_o::Array, F)
    a = @. 1/sqrt(1+μ_o^2*(1/F^2-1))
    result = (μ_o./F) .* a
    return result
end

function _P_obs(k_o, μ_o, q_par, q_perp, Int_Mono, Int_Quad, Int_Hexa)
    F = q_par/q_perp
    k_t = _k_true(k_o, μ_o, q_perp, F)
    μ_t = _μ_true(μ_o, F)

    return _Pkμ(k_t, μ_t, Int_Mono, Int_Quad, Int_Hexa)/(q_par*q_perp^2)
end

function interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    #extrapolation might introduce some errors ar high k, when q << 1.
    #maybe we should implement a log extrapolation?
    Int_Mono = AkimaInterpolation(Mono_array, k_grid; extrapolation = ExtrapolationType.Extension)
    Int_Quad = AkimaInterpolation(Quad_array, k_grid; extrapolation = ExtrapolationType.Extension)
    Int_Hexa = AkimaInterpolation(Hexa_array, k_grid; extrapolation = ExtrapolationType.Extension)
    return Int_Mono, Int_Quad, Int_Hexa
end

function k_projection(k_projection, Mono_array, Quad_array, Hexa_array, k_grid)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return int_Mono.(k_projection), int_Quad.(k_projection), int_Hexa.(k_projection)
end

"""
    apply_AP_check(k_grid::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array,
    q_par, q_perp)
Given the Monopole, the Quadrupole, the Hexadecapole, and the k-grid, this function apply
the AP effect using the Gauss-Kronrod adaptive quadrature. Precise, but expensive, function.
Mainly used for check and debugging purposes.
"""
function apply_AP_check(k_grid::Array, Mono_array::Array, Quad_array::Array,
    Hexa_array::Array, q_par, q_perp)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return apply_AP_check(k_grid, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

function apply_AP_check(k_grid, int_Mono::DataInterpolations.AbstractInterpolation,
    int_Quad::DataInterpolations.AbstractInterpolation,
    int_Hexa::DataInterpolations.AbstractInterpolation, q_par, q_perp)
    nk = length(k_grid)
    result = zeros(3, nk)
    ℓ_array = [0,2,4]
    for i in 1:nk # TODO: use enumerate(k_grid)
        for (ℓ_idx, myℓ) in enumerate(ℓ_array)
            result[ℓ_idx, i] = (2*myℓ+1)*quadgk(x -> Pl(x, myℓ)*_P_obs(k_grid[i], x, q_par,
            q_perp, int_Mono, int_Quad, int_Hexa), 0, 1, rtol=1e-12)[1]
        end
    end
    return result[1,:], result[2,:], result[3,:]
end

function _k_grid_over_nl(k_grid, k_nl)
    return @. (k_grid/k_nl)^2
 end

function q_par_perp(z, cosmo_mcmc::AbstractCosmology, cosmo_ref::AbstractCosmology)
    E_ref  = _E_z(z, cosmo_ref)
    E_mcmc = _E_z(z, cosmo_mcmc)

    d̃A_ref  = _d̃A_z(z, cosmo_ref)
    d̃A_mcmc = _d̃A_z(z, cosmo_mcmc)

    q_perp  = d̃A_mcmc/d̃A_ref
    q_par = E_ref/E_mcmc

    return q_par, q_perp
end

function Pk_recon(mono::Matrix, quad::Matrix, hexa::Matrix, l0, l2, l4)
    return mono.*l0' .+ quad.*l2' + hexa.*l4'
end

"""
    apply_AP(k_grid::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par,
    q_perp)
Given the Monopole, the Quadrupole, the Hexadecapole, and the k-grid, this function apply
the AP effect using the Gauss-Lobatto quadrature. Fast but accurate,  well tested against
adaptive Gauss-Kronrod integration.
"""
function apply_AP(k::Array, mono::Array, quad::Array, hexa::Array, q_par, q_perp;
    n_GL_points=8)
    nk = length(k)
    nodes, weights = gausslobatto(n_GL_points*2)
    #since the integrand is symmetric, we are gonna use only half of the points
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    F = q_par/q_perp

    k_t = k_true(k, μ_nodes, q_perp, F)

    μ_t = μ_true(μ_nodes, F)

    Pl0_t = _legendre_0.(μ_t)
    Pl2_t = _legendre_2.(μ_t)
    Pl4_t = _legendre_4.(μ_t)

    Pl0 = _legendre_0.(μ_nodes).*μ_weights.*(2*0+1)
    Pl2 = _legendre_2.(μ_nodes).*μ_weights.*(2*2+1)
    Pl4 = _legendre_4.(μ_nodes).*μ_weights.*(2*4+1)

    new_mono = reshape(_akima_spline(mono, k, k_t), nk, n_GL_points)
    new_quad = reshape(_akima_spline(quad, k, k_t), nk, n_GL_points)
    new_hexa = reshape(_akima_spline(hexa, k, k_t), nk, n_GL_points)

    Pkμ = Pk_recon(new_mono, new_quad, new_hexa, Pl0_t, Pl2_t, Pl4_t)./(q_par*q_perp^2)

    pippo_0 = Pkμ * Pl0
    pippo_2 = Pkμ * Pl2
    pippo_4 = Pkμ * Pl4

    return pippo_0, pippo_2, pippo_4
end

function window_convolution(W::Array{T, 4}, v::Matrix) where T
    return @tullio C[i,k] := W[i,j,k,l] * v[j,l]
end
