@non_differentiable LinRange(a, b, n)
@non_differentiable _transformed_weights(quadrature_rule, order, a, b)
@non_differentiable gausslobatto(n)

# Adjoint for 2D/1D case: matrix-vector multiplication
@adjoint function window_convolution(W::AbstractMatrix, v::AbstractVector)
    C = window_convolution(W, v)
    function window_convolution_2d_pullback(C̄)
        ∂W = C̄ * v'  # outer product
        ∂v = W' * C̄  # matrix-vector product
        return (∂W, ∂v)
    end
    return (C, window_convolution_2d_pullback)
end

# Adjoint for 4D/2D case: tensor contraction
@adjoint function window_convolution(W, v)
    C = window_convolution(W, v)
    function window_convolution_pullback(C̄)
        # Remove @thunk wrappers for direct computation - the @tullio operations
        # are already optimized and the overhead of thunking is not beneficial here
        ∂W = first_rule(C̄, v)
        ∂v = second_rule(C̄, W)
        return (∂W, ∂v)
    end
    return (C, window_convolution_pullback)
end

function first_rule(dC, v)
    @tullio dW[i, j, k, l] := dC[i, k] * v[j, l]
    return dW
end

function second_rule(dC, W)
    @tullio dv[j, l] := dC[i, k] * W[i, j, k, l]
    return dv
end

@adjoint function _akima_slopes(u::AbstractVector, t::AbstractVector)                   # length n-1
    n = length(u)
    dt = diff(t)                     # length n-1
    m = zeros(eltype(u), n + 3)

    m[3:(n+1)] .= diff(u) ./ dt      # interior slopes
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[n+2] = 2m[n+1] - m[n]
    m[n+3] = 2m[n+2] - m[n+1]

    function pullback(Δm)
        # Ensure gm is a mutable array - handles Fill arrays and other immutable types
        gm = collect(Δm)             # running adjoint of m

        # --- extrapolation terms: do them in *reverse* program order ---

        # m[n+3] = 2m[n+2] - m[n+1]
        gm[n+2] += 2gm[n+3]
        gm[n+1] -= gm[n+3]

        # m[n+2] = 2m[n+1] - m[n]
        gm[n+1] += 2gm[n+2]
        gm[n] -= gm[n+2]

        # m[1] = 2m[2] - m[3]
        gm[2] += 2gm[1]
        gm[3] -= gm[1]

        # m[2] = 2m[3] - m[4]
        gm[3] += 2gm[2]
        gm[4] -= gm[2]

        # --- back-prop through the interior slopes --------------------
        sm_bar = gm[3:(n+1)]         # ∂L/∂((u[i+1]-u[i])/dt[i])

        δu = zero(u)
        δt = zero(t)

        @inbounds for i in 1:n-1
            g = sm_bar[i]
            invdt = 1 / dt[i]

            # w.r.t. u
            δu[i] -= g * invdt
            δu[i+1] += g * invdt

            # w.r.t. t      d/dt ( (u₊ − u)/dt ) = −(u₊−u)/dt²  on both endpoints
            diffu = u[i+1] - u[i]
            invdt2 = invdt^2
            δt[i] += g * diffu * invdt2
            δt[i+1] -= g * diffu * invdt2
        end

        return (δu, δt)
    end

    return m, pullback
end

@adjoint function _akima_coefficients(t, m)
    n = length(t)
    dt = diff(t)

    # Forward computation - must match utils.jl implementation
    dm = abs.(diff(m))
    f1 = dm[3:(n+2)]
    f2 = dm[1:n]
    f12 = f1 + f2
    b = (m[4:end] .+ m[1:(end-3)]) ./ 2  # Average slope (fallback)

    # Handle division by zero for constant/linear segments
    eps_akima = eps(eltype(f12)) * 100
    use_weighted = f12 .> eps_akima
    for i in eachindex(f12)
        if use_weighted[i]
            b[i] = (f1[i] * m[i+1] + f2[i] * m[i+2]) / f12[i]
        end
    end

    c = (3 .* m[3:(end-2)] .- 2 .* b[1:(end-1)] .- b[2:end]) ./ dt
    d = (b[1:(end-1)] .+ b[2:end] .- 2 .* m[3:(end-2)]) ./ dt .^ 2

    function _akima_coefficients_pullback(Δ)
        Δb, Δc, Δd = Δ

        # Pre-allocate gradient arrays once and reuse - major optimization
        ∂t = zeros(eltype(t), length(t))
        ∂m = zeros(eltype(m), length(m))
        # Pre-allocate b gradient accumulator to avoid multiple zero arrays
        ∂b_accum = zeros(eltype(b), length(b))

        # Cache commonly used values for efficiency
        dt_inv_sq = @. 1.0 / dt^2  # Precompute 1/dt² to avoid repeated division
        dt_inv = @. 1.0 / dt       # Precompute 1/dt

        # Pullback through d computation - optimized conditional handling
        if Δd !== nothing
            # d = (b[1:(end - 1)] .+ b[2:end] .- 2 .* m[3:(end - 2)]) ./ dt.^2
            # Vectorized gradient computation for better performance
            @. ∂b_accum[1:(end-1)] += Δd * dt_inv_sq
            @. ∂b_accum[2:end] += Δd * dt_inv_sq
            @. ∂m[3:(end-2)] -= 2.0 * Δd * dt_inv_sq

            # Optimized t gradient computation using cached values
            ∂dt_from_d = @. -2.0 * Δd * (b[1:(end-1)] + b[2:end] - 2.0 * m[3:(end-2)]) * dt_inv_sq / dt
            @. ∂t[1:(end-1)] -= ∂dt_from_d
            @. ∂t[2:end] += ∂dt_from_d
        end

        # Pullback through c computation - optimized
        if Δc !== nothing
            # c = (3 .* m[3:(end - 2)] .- 2 .* b[1:(end - 1)] .- b[2:end]) ./ dt
            @. ∂m[3:(end-2)] += 3.0 * Δc * dt_inv
            @. ∂b_accum[1:(end-1)] -= 2.0 * Δc * dt_inv
            @. ∂b_accum[2:end] -= Δc * dt_inv

            # Optimized t gradient computation
            ∂dt_from_c = @. -Δc * (3.0 * m[3:(end-2)] - 2.0 * b[1:(end-1)] - b[2:end]) * dt_inv^2
            @. ∂t[1:(end-1)] -= ∂dt_from_c
            @. ∂t[2:end] += ∂dt_from_c
        end

        # Combine b gradients from d and c with input gradients
        if Δb !== nothing
            @. ∂b_accum += Δb
        end

        # Pullback through b computation - only if we have b gradients to propagate
        if any(!iszero, ∂b_accum)
            # Need to handle two cases:
            # - When use_weighted[i]: b[i] = (f1[i] * m[i+1] + f2[i] * m[i+2]) / f12[i]
            # - When !use_weighted[i]: b[i] = (m[i+3] + m[i]) / 2

            ∂f1 = zeros(eltype(f1), length(f1))
            ∂f2 = zeros(eltype(f2), length(f2))
            ∂f12 = zeros(eltype(f12), length(f12))

            for i in eachindex(use_weighted)
                if use_weighted[i]
                    # Weighted average case
                    f12_inv_i = 1.0 / f12[i]
                    ∂f1[i] += ∂b_accum[i] * m[i+1] * f12_inv_i
                    ∂f2[i] += ∂b_accum[i] * m[i+2] * f12_inv_i
                    ∂m[i+1] += ∂b_accum[i] * f1[i] * f12_inv_i
                    ∂m[i+2] += ∂b_accum[i] * f2[i] * f12_inv_i
                    ∂f12[i] += -∂b_accum[i] * (f1[i] * m[i+1] + f2[i] * m[i+2]) * f12_inv_i^2
                else
                    # Simple average case: b[i] = (m[i+3] + m[i]) / 2
                    ∂m[i+3] += ∂b_accum[i] / 2
                    ∂m[i] += ∂b_accum[i] / 2
                end
            end

            # f12 = f1 + f2
            @. ∂f1 += ∂f12
            @. ∂f2 += ∂f12

            # Pre-allocate ∂dm only once and reuse for both f1 and f2 gradients
            ∂dm = zeros(eltype(dm), length(dm))
            @. ∂dm[3:(n+2)] += ∂f1  # f1 = dm[3:(n + 2)]
            @. ∂dm[1:n] += ∂f2      # f2 = dm[1:n]

            # dm = abs.(diff(m)) - optimized sign computation
            diff_m = diff(m)
            ∂diff_m = @. ∂dm * sign(diff_m)

            # diff(m) pullback - vectorized
            @. ∂m[1:(end-1)] -= ∂diff_m
            @. ∂m[2:end] += ∂diff_m
        end

        return (∂t, ∂m)
    end

    return (b, c, d), _akima_coefficients_pullback
end

@adjoint function _akima_eval(u, t, b, c, d, tq::AbstractArray)
    # Forward pass - Replace map() with pre-allocated loop for better performance
    n_query = length(tq)
    # Promote ALL input types for proper ForwardDiff support
    T = promote_type(eltype(u), eltype(t), eltype(b), eltype(c), eltype(d), eltype(tq))
    results = zeros(T, n_query)

    # Vectorized forward evaluation with better memory locality
    @inbounds for i in eachindex(tq)
        idx = _akima_find_interval(t, tq[i])
        wj = tq[i] - t[idx]
        # Horner's method evaluation: ((d*w + c)*w + b)*w + u
        results[i] = ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]
    end

    function pullback(ȳ)
        # Pre-allocate all gradients once for better memory efficiency
        ū_total = zero(u)
        t̄_total = zero(t)
        b̄_total = zero(b)
        c̄_total = zero(c)
        d̄_total = zero(d)
        tq̄ = similar(tq, promote_type(eltype(ȳ), eltype(tq)))

        # Optimized gradient accumulation loop with better SIMD potential
        @inbounds for i in eachindex(tq)
            ȳ_i = ȳ[i]
            if !iszero(ȳ_i)  # Skip computation for zero gradients
                idx = _akima_find_interval(t, tq[i])
                wj = tq[i] - t[idx]

                # Compute polynomial derivative efficiently
                # For f(w) = d*w³ + c*w² + b*w + u, f'(w) = 3*d*w² + 2*c*w + b
                wj_sq = wj * wj
                dwj = 3 * d[idx] * wj_sq + 2 * c[idx] * wj + b[idx]

                # Accumulate gradients efficiently - avoiding redundant array indexing
                ū_total[idx] += ȳ_i
                t̄_total[idx] -= ȳ_i * dwj
                tq̄[i] = ȳ_i * dwj
                b̄_total[idx] += ȳ_i * wj
                c̄_total[idx] += ȳ_i * wj_sq
                d̄_total[idx] += ȳ_i * wj * wj_sq  # wj³
            else
                tq̄[i] = zero(eltype(tq̄))
            end
        end

        return ū_total, t̄_total, b̄_total, c̄_total, d̄_total, tq̄
    end

    return results, pullback
end

@adjoint function _akima_slopes(u::AbstractMatrix, t)
    n, n_cols = size(u)
    dt = diff(t)
    m = zeros(promote_type(eltype(u), eltype(t)), n + 3, n_cols)

    # Forward pass matches utils.jl matrix implementation
    for col in 1:n_cols
        m[3:(end-2), col] .= diff(view(u, :, col)) ./ dt
        m[2, col] = 2m[3, col] - m[4, col]
        m[1, col] = 2m[2, col] - m[3, col]
        m[n+2, col] = 2m[n+1, col] - m[n, col]
        m[n+3, col] = 2m[n+2, col] - m[n+1, col]
    end

    function pullback_matrix(Δm)
        ∂u = zero(u)
        ∂t = zero(t)

        # Process each column using the vector adjoint logic
        for col in 1:n_cols
            Δm_col = collect(Δm[:, col])

            # Apply vector adjoint logic for this column
            # Extrapolation terms in reverse order
            Δm_col[n+2] += 2Δm_col[n+3]
            Δm_col[n+1] -= Δm_col[n+3]
            Δm_col[n+1] += 2Δm_col[n+2]
            Δm_col[n] -= Δm_col[n+2]
            Δm_col[2] += 2Δm_col[1]
            Δm_col[3] -= Δm_col[1]
            Δm_col[3] += 2Δm_col[2]
            Δm_col[4] -= Δm_col[2]

            # Interior slopes gradient
            sm_bar = Δm_col[3:(n+1)]

            @inbounds for i in 1:n-1
                g = sm_bar[i]
                invdt = 1 / dt[i]

                # w.r.t. u
                ∂u[i, col] -= g * invdt
                ∂u[i+1, col] += g * invdt

                # w.r.t. t
                diffu = u[i+1, col] - u[i, col]
                invdt2 = invdt^2
                ∂t[i] += g * diffu * invdt2
                ∂t[i+1] -= g * diffu * invdt2
            end
        end

        return (∂u, ∂t)
    end

    return m, pullback_matrix
end

@adjoint function _akima_coefficients(t, m::AbstractMatrix)
    # Optimized matrix version without recursive Zygote calls

    n = length(t)
    n_cols = size(m, 2)
    dt = diff(t)
    eps_akima = eps(eltype(m)) * 100

    # Pre-allocate coefficient arrays
    b = zeros(eltype(m), n, n_cols)
    c = zeros(eltype(m), n - 1, n_cols)
    d = zeros(eltype(m), n - 1, n_cols)
    use_weighted = falses(n, n_cols)  # Track which indices use weighted interpolation

    # Forward computation for each column
    for col in 1:n_cols
        b[:, col] = (view(m, 4:(n+3), col) .+ view(m, 1:n, col)) ./ 2

        dm = abs.(diff(view(m, :, col)))
        f1 = view(dm, 3:(n+2))
        f2 = view(dm, 1:n)
        f12 = f1 .+ f2

        for i in 1:n
            if f12[i] > eps_akima
                b[i, col] = (f1[i] * m[i+1, col] + f2[i] * m[i+2, col]) / f12[i]
                use_weighted[i, col] = true
            end
        end

        c[:, col] = (3 .* view(m, 3:(n+1), col) .- 2 .* view(b, 1:(n-1), col) .- view(b, 2:n, col)) ./ dt
        d[:, col] = (view(b, 1:(n-1), col) .+ view(b, 2:n, col) .- 2 .* view(m, 3:(n+1), col)) ./ dt .^ 2
    end

    function pullback_matrix_coeffs(Δ)
        Δb, Δc, Δd = Δ
        ∂t = zeros(eltype(t), n)
        ∂m = zeros(eltype(m), n + 3, n_cols)

        for col in 1:n_cols
            dm = abs.(diff(view(m, :, col)))
            f1 = view(dm, 3:(n+2))
            f2 = view(dm, 1:n)
            f12 = f1 .+ f2

            # Gradients from c
            for i in 1:(n-1)
                ∂m[i+2, col] += Δc[i, col] * 3 / dt[i]
                Δb[i, col] -= Δc[i, col] * 2 / dt[i]
                Δb[i+1, col] -= Δc[i, col] / dt[i]

                numerator_c = 3 * m[i+2, col] - 2 * b[i, col] - b[i+1, col]
                ∂t[i+1] -= Δc[i, col] * numerator_c / dt[i]^2
                ∂t[i] += Δc[i, col] * numerator_c / dt[i]^2
            end

            # Gradients from d
            for i in 1:(n-1)
                Δb[i, col] += Δd[i, col] / dt[i]^2
                Δb[i+1, col] += Δd[i, col] / dt[i]^2
                ∂m[i+2, col] -= Δd[i, col] * 2 / dt[i]^2

                numerator_d = b[i, col] + b[i+1, col] - 2 * m[i+2, col]
                ∂t[i+1] -= Δd[i, col] * 2 * numerator_d / dt[i]^3
                ∂t[i] += Δd[i, col] * 2 * numerator_d / dt[i]^3
            end

            # Gradients through b (conditional)
            for i in 1:n
                if use_weighted[i, col]
                    ∂m[i+1, col] += Δb[i, col] * f1[i] / f12[i]
                    ∂m[i+2, col] += Δb[i, col] * f2[i] / f12[i]

                    df1 = Δb[i, col] * (m[i+1, col] - b[i, col]) / f12[i]
                    sign_f1 = sign(m[i+3, col] - m[i+2, col])
                    ∂m[i+3, col] += df1 * sign_f1
                    ∂m[i+2, col] -= df1 * sign_f1

                    df2 = Δb[i, col] * (m[i+2, col] - b[i, col]) / f12[i]
                    sign_f2 = sign(m[i+1, col] - m[i, col])
                    ∂m[i+1, col] += df2 * sign_f2
                    ∂m[i, col] -= df2 * sign_f2
                else
                    ∂m[i+3, col] += Δb[i, col] / 2
                    ∂m[i, col] += Δb[i, col] / 2
                end
            end
        end

        return (∂t, ∂m)
    end

    return (b, c, d), pullback_matrix_coeffs
end

@adjoint function _akima_eval(u::AbstractMatrix, t, b::AbstractMatrix, c::AbstractMatrix,
                               d::AbstractMatrix, tq::AbstractArray)
    n_query = length(tq)
    n_cols = size(u, 2)
    # Promote ALL input types for proper ForwardDiff support
    T = promote_type(eltype(u), eltype(t), eltype(b), eltype(c), eltype(d), eltype(tq))
    results = zeros(T, n_query, n_cols)

    # Forward pass using optimized matrix implementation
    @inbounds for i in 1:n_query
        idx = _akima_find_interval(t, tq[i])
        wj = tq[i] - t[idx]

        @simd for col in 1:n_cols
            results[i, col] = ((d[idx, col] * wj + c[idx, col]) * wj + b[idx, col]) * wj + u[idx, col]
        end
    end

    function pullback_matrix_eval(ȳ)
        ū = zero(u)
        t̄ = zero(t)
        b̄ = zero(b)
        c̄ = zero(c)
        d̄ = zero(d)
        tq̄ = zeros(promote_type(eltype(ȳ), eltype(tq)), n_query)

        # Compute gradients for all columns
        @inbounds for i in 1:n_query
            idx = _akima_find_interval(t, tq[i])
            wj = tq[i] - t[idx]
            wj_sq = wj * wj
            wj_cb = wj * wj_sq

            tq̄_accum = zero(eltype(tq̄))
            t̄_accum = zero(eltype(t̄))

            @simd for col in 1:n_cols
                ȳ_ic = ȳ[i, col]
                if !iszero(ȳ_ic)
                    # Polynomial derivative: f'(w) = 3*d*w² + 2*c*w + b
                    dwj = 3 * d[idx, col] * wj_sq + 2 * c[idx, col] * wj + b[idx, col]

                    ū[idx, col] += ȳ_ic
                    t̄_accum -= ȳ_ic * dwj
                    tq̄_accum += ȳ_ic * dwj
                    b̄[idx, col] += ȳ_ic * wj
                    c̄[idx, col] += ȳ_ic * wj_sq
                    d̄[idx, col] += ȳ_ic * wj_cb
                end
            end

            t̄[idx] += t̄_accum
            tq̄[i] = tq̄_accum
        end

        return ū, t̄, b̄, c̄, d̄, tq̄
    end

    return results, pullback_matrix_eval
end
