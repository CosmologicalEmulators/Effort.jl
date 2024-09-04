function _Pkμ(k, μ, Int_Mono, Int_Quad, Int_Hexa)
    #@info Int_Hexa(k)
    return Int_Mono(k)*_legendre_0(μ) + Int_Quad(k)*_legendre_2(μ) + Int_Hexa(k)*_legendre_4(μ)
end

function _k_true(k_o, μ_o, q_perp, F)
    return @. k_o/q_perp*sqrt(1+μ_o^2*(1/F^2-1))
end

function _μ_true(μ_o, F)
    return μ_o/F/sqrt(1+μ_o^2*(1/F^2-1))
end

function k_true(k_o::Array, μ_o::Array, q_perp, F)
    @tullio result[i,j] := k_o[i]/q_perp*sqrt(1+μ_o[j]^2*(1/F^2-1))
    return vec(result)
end

function μ_true(μ_o::Array, F)
    @tullio result[i] := μ_o[i]/F/sqrt(1. +μ_o[i]^2*(1/F^2-1))
    return result
end

function _P_obs(k_o, μ_o, q_par, q_perp, Int_Mono, Int_Quad, Int_Hexa)
    F = q_par/q_perp
    k_t = _k_true(k_o, μ_o, q_perp, F)
    μ_t = _μ_true(μ_o, F)

    _Pkμ(k_t, μ_t, Int_Mono, Int_Quad, Int_Hexa)/(q_par*q_perp^2)

    return _Pkμ(k_t, μ_t, Int_Mono, Int_Quad, Int_Hexa)/(q_par*q_perp^2)
end

function interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    #extrapolation might introduce some errors ar high k, when q << 1.
    #maybe we should implement a log extrapolation?
    Int_Mono = QuadraticSpline(Mono_array, k_grid; extrapolate = true)
    Int_Quad = QuadraticSpline(Quad_array, k_grid; extrapolate = true)
    Int_Hexa = QuadraticSpline(Hexa_array, k_grid; extrapolate = true)
    return Int_Mono, Int_Quad, Int_Hexa
end

function k_projection(k_projection, Mono_array, Quad_array, Hexa_array, k_grid)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return int_Mono.(k_projection), int_Quad.(k_projection), int_Hexa.(k_projection)
end

function apply_AP_check(k_grid, int_Mono::QuadraticSpline, int_Quad::QuadraticSpline,
    int_Hexa::QuadraticSpline, q_par, q_perp)
    nk = length(k_grid)
    result = zeros(3, nk)
    ℓ_array = [0,2,4]
    for i in 1:nk # TODO: use enumerate(k_grid)
        for (ℓ_idx, myℓ) in enumerate(ℓ_array)
            result[ℓ_idx, i] = (2*myℓ+1)*quadgk(x -> Pl(x, myℓ)*_P_obs(k_grid[i], x, q_par,
            q_perp, int_Mono, int_Quad, int_Hexa), 0, 1, rtol=1e-12)[1]
        end
    end
    return result
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

function _stoch_obs(k_o, μ_o, q_par, q_perp, n_bar, cϵ0, cϵ1, cϵ2, k_nl)
    F = q_par/q_perp
    k_t = _k_true(k_o, μ_o, q_perp, F)
    μ_t = _μ_true(μ_o, F)
    return _stoch_kμ(k_t, μ_t, n_bar, cϵ0, cϵ1, cϵ2, k_nl)/(q_par*q_perp^2)
end

function _k_grid_over_nl(k_grid, k_nl)
    return @. (k_grid/k_nl)^2
 end

 function _stoch_kμ(k_grid, μ, n_bar, cϵ0, cϵ1, cϵ2, k_nl)
    return (cϵ0 * _legendre_0(μ) .+ Effort._k_grid_over_nl(k_grid, k_nl) .* (cϵ1 * _legendre_0(μ) +
            cϵ2 * _legendre_2(μ)) ) ./ n_bar
end

function get_stochs_AP(k_grid, q_par, q_perp, n_bar, cϵ0, cϵ1, cϵ2; k_nl = 0.7, n_GL_points = 18)
    nk = length(k_grid)
    #TODO: check that the extrapolation does not create problems. Maybe logextrap?
    nodes, weights = @memoize gausslobatto(n_GL_points*2)
    #since the integrand is symmetric, we are gonna use only half of the points
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    result = zeros(2, nk)

    Pl_array = zeros(2, n_GL_points)
    Pl_array[1,:] .= _legendre_0.(μ_nodes)
    Pl_array[2,:] .= _legendre_2.(μ_nodes)

    temp = zeros(n_GL_points, nk)

    for i in 1:n_GL_points
        temp[i,:] = _stoch_obs(k_grid, μ_nodes[i], q_par, q_perp, n_bar, cϵ0, cϵ1, cϵ2,
                               k_nl)
    end

    multipole_weight = [2*0+1, 2*2+1]

    @turbo for i in 1:2
        for j in 1:nk
            for k in 1:n_GL_points
                result[i, j] += μ_weights[k] * temp[k,j] * Pl_array[i, k] *
                                multipole_weight[i]
            end
        end
    end

    return result
end

function q_par_perp(z, ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)
    E_ref  = _E_z(z, ΩM_ref, w0_ref, wa_ref)
    E_true = _E_z(z, ΩM_true, w0_true, wa_true)

    d̃A_ref  = _d̃A_z(z, ΩM_ref, w0_ref, wa_ref)
    d̃A_true = _d̃A_z(z, ΩM_true, w0_true, wa_true)

    q_perp = E_true/E_ref
    q_par  = d̃A_ref/d̃A_true
    return q_par, q_perp
end

function q_par_perp(z, ΩM_ref, w0_ref, wa_ref, E_true, d̃A_true)
    E_ref  = _E_z(z, ΩM_ref, w0_ref, wa_ref)

    d̃A_ref  = _d̃A_z(z, ΩM_ref, w0_ref, wa_ref)

    q_perp = E_true/E_ref
    q_par  = d̃A_ref/d̃A_true
    return q_perp, q_par
end

function apply_AP(k_grid, Mono_array::Array, Quad_array::Array, Hexa_array::Array, z, ΩM_ref,
    w0_ref, wa_ref, ΩM_true, w0_true, wa_true)

    q_par, q_perp  = q_par_perp(z, ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return apply_AP(k_grid, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

function apply_AP(k_grid_AP, k_grid, Mono_array::Array, Quad_array::Array, Hexa_array::Array, z,
    ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)

    q_par, q_perp  = q_par_perp(z, ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)

    return apply_AP(k_grid_AP, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

function Pk_recon(mono, quad, hexa, l0, l2, l4)
    @tullio Pkμ[i,j] := mono[i,j]*l0[j] + quad[i,j]*l2[j] + hexa[i,j]*l4[j]
     return Pkμ
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

    new_mono = reshape(_quadratic_spline(mono, k, k_t), nk, n_GL_points)
    new_quad = reshape(_quadratic_spline(quad, k, k_t), nk, n_GL_points)
    new_hexa = reshape(_quadratic_spline(hexa, k, k_t), nk, n_GL_points)

    Pkμ = Pk_recon(new_mono, new_quad, new_hexa, Pl0_t, Pl2_t, Pl4_t)./(q_par*q_perp^2)

    pippo_0 = Pkμ * Pl0
    pippo_2 = Pkμ * Pl2
    pippo_4 = Pkμ * Pl4
    result = hcat(pippo_0, pippo_2, pippo_4)'

    return result
end

function window_convolution(W,v)
    return @tullio C[i,k] := W[i,j,k,l] * v[j,l]
end
