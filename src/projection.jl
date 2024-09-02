function _Pkμ(k, μ, Int_Mono, Int_Quad, Int_Hexa)
    return Int_Mono(k)*Pl(μ, 0) + Int_Quad(k)*Pl(μ, 2) + Int_Hexa(k)*Pl(μ, 4)
end

function _k_true(k_o, μ_o, q_perp, F)
    return @. k_o/q_perp*sqrt(1+μ_o^2*(1/F^2-1))
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

function _mygemmavx(A, B, C)
    Dm = zero(eltype(C))
    @tturbo for n ∈ axes(A,1)
        Dm += A[n] * B[n] * C[n]
    end
    return Dm
end

function _mygemm(A, B, C)
    Dm = zero(eltype(C))
    for n ∈ axes(A,1)
        Dm += A[n] * B[n] * C[n]
    end
    return Dm
end

function apply_AP(k_grid, int_Mono::QuadraticSpline, int_Quad::QuadraticSpline, int_Hexa::QuadraticSpline,
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
    return (cϵ0 * Pl(μ, 0) .+ Effort._k_grid_over_nl(k_grid, k_nl) .* (cϵ1 * Pl(μ, 0) +
            cϵ2 * Pl(μ, 2)) ) ./ n_bar
end

function get_stochs_AP(k_grid, q_par, q_perp, n_bar, cϵ0, cϵ1, cϵ2; k_nl = 0.7, n_GL_points = 4)
    nk = length(k_grid)
    #TODO: check that the extrapolation does not create problems. Maybe logextrap?
    nodes, weights = @memoize gausslobatto(n_GL_points*2)
    #since the integrand is symmetric, we are gonna use only half of the points
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    result = zeros(2, nk)

    Pl_array = zeros(2, n_GL_points)
    Pl_array[1,:] .= Pl.(μ_nodes, 0)
    Pl_array[2,:] .= Pl.(μ_nodes, 2)

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

"""
function window_convolution(W,v)
    a,b,c,d = size(W)
    result = zeros(a,c)
    window_convolution!(result,W,v)
    return result
end

function window_convolution!(result, W, v)
    @turbo for i ∈ axes(W,1), k ∈ axes(W,3)
        c = zero(eltype(v))
        for l ∈ axes(W,4)
            for j ∈ axes(W,2)
                c += W[i,j,k,l] * v[j,l]
            end
        end
        result[i,k] = c
    end
end
"""

function window_convolution(W,v)
    return @tullio C[i,k] := W[i,j,k,l] * v[j,l]
end
