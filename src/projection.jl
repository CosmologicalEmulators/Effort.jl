function _Pkμ(k, μ, Int_Mono, Int_Quad, Int_Hexa)
    return Int_Mono(k)*Pl(μ, 0) + Int_Quad(k)*Pl(μ, 2) + Int_Hexa(k)*Pl(μ, 4)
end

function _k_true(k_o, μ_o, q_perp, F)
    return k_o/q_perp*sqrt(1+μ_o^2*(1/F^2-1))
end

function _μ_true(μ_o, F)
    return μ_o/F/sqrt(1+μ_o^2*(1/F^2-1))
end

function _P_obs(k_o, μ_o, q_par, q_perp, Int_Mono, Int_Quad, Int_Hexa)
    F = q_par/q_perp
    k_t = _k_true(k_o, μ_o, q_perp, F)
    μ_t = _μ_true(μ_o, F)
    return _Pkμ(k_t, μ_t, Int_Mono, Int_Quad, Int_Hexa)/(q_par*q_perp^2)
end

function interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    Int_Mono = CubicSpline(Mono_array, k_grid)
    Int_Quad = CubicSpline(Quad_array, k_grid)
    Int_Hexa = CubicSpline(Hexa_array, k_grid)
    return Int_Mono, Int_Quad, Int_Hexa
end

function k_projection(k_projection, Mono_array, Quad_array, Hexa_array, k_grid)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return int_Mono.(k_projection), int_Quad.(k_projection), int_Hexa.(k_projection)
end

function apply_AP_check(k_grid, int_Mono::CubicSpline, int_Quad::CubicSpline,
    int_Hexa::CubicSpline, q_par, q_perp)
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

function _mygemmavx(A, B, C)
    Dm = zero(eltype(C))
    @turbo for n ∈ axes(A,1)
        Dm += A[n] * B[n] * C[n]
    end
    return Dm
end

function apply_AP(k_grid, int_Mono::CubicSpline, int_Quad::CubicSpline, int_Hexa::CubicSpline,
    q_par, q_perp)
    nk = length(k_grid)
    n_GL_points = 5
    #TODO: check that the extrapolation does not create problems. Maybe logextrap?
    nodes, weights = @memoize gausslobatto(n_GL_points*2)
    #since the integrand is symmetric, we are gonna use only half of the points
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    result = zeros(3, nk)

    Pl_0 = Pl.(μ_nodes, 0)
    Pl_2 = Pl.(μ_nodes, 2)
    Pl_4 = Pl.(μ_nodes, 4)

    temp = zeros(n_GL_points)

    for (k_idx, myk) in enumerate(k_grid)
        for j in 1:n_GL_points
            temp[j] = _P_obs(myk, μ_nodes[j], q_par, q_perp, int_Mono, int_Quad,
            int_Hexa)
        end
        #we do not divided by 2 since we are using only half of the points and the result
        #should be multiplied by 2
        result[1, k_idx] = (2*0+1)*_mygemmavx(μ_weights, temp, Pl_0)
        result[2, k_idx] = (2*2+1)*_mygemmavx(μ_weights, temp, Pl_2)
        result[3, k_idx] = (2*4+1)*_mygemmavx(μ_weights, temp, Pl_4)
    end
    return result
end

function q_perp_par(z, ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)
    E_ref  = _E_z(z, ΩM_ref, w0_ref, wa_ref)
    E_true = _E_z(z, ΩM_true, w0_true, wa_true)

    d̃A_ref  = _d̃A_z(z, ΩM_ref, w0_ref, wa_ref)
    d̃A_true = _d̃A_z(z, ΩM_true, w0_true, wa_true)

    q_perp = E_ref/E_true
    q_par  = d̃A_true/d̃A_ref
    return q_perp, q_par
end

function apply_AP(k_grid, Mono_array::Array, Quad_array::Array, Hexa_array::Array, z, ΩM_ref,
    w0_ref, wa_ref, ΩM_true, w0_true, wa_true)

    q_perp, q_par  = q_perp_par(z, ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return apply_AP(k_grid, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

function apply_AP(k_grid_AP, k_grid, Mono_array::Array, Quad_array::Array, Hexa_array::Array, z,
    ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)

    q_perp, q_par  = q_perp_par(z, ΩM_ref, w0_ref, wa_ref, ΩM_true, w0_true, wa_true)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)

    return apply_AP(k_grid_AP, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

"""
    apply_AP(k_grid::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par,
    q_perp)
Given the Monopole, the Quadrupole, the Hexadecapole, and the k-grid, this function apply
the AP effect using the Gauss-Lobatto quadrature. Fast but accurate,  well tested against
adaptive Gauss-Kronrod integration.
"""
function apply_AP(k_grid, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par,
    q_perp)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    return apply_AP(k_grid, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

function apply_AP(k_grid_AP, k_interp, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par, q_perp)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_interp)
    return apply_AP(k_grid_AP, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end
