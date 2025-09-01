@non_differentiable LinRange(a, b, n)
@non_differentiable _transformed_weights(quadrature_rule, order, a, b)
@non_differentiable gausslobatto(n)

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

    # Forward computation
    dm = abs.(diff(m))
    f1 = dm[3:(n+2)]
    f2 = dm[1:n]
    f12 = f1 + f2
    b = (f1 .* m[2:(end-2)] .+ f2 .* m[3:(end-1)]) ./ f12
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
            # b = (f1 .* m[2:(end-2)] .+ f2 .* m[3:(end-1)]) ./ f12
            # Vectorized computation avoiding intermediate arrays
            f12_inv = @. 1.0 / f12  # Precompute reciprocal
            ∂f1 = @. ∂b_accum * m[2:(end-2)] * f12_inv
            ∂f2 = @. ∂b_accum * m[3:(end-1)] * f12_inv
            @. ∂m[2:(end-2)] += ∂b_accum * f1 * f12_inv
            @. ∂m[3:(end-1)] += ∂b_accum * f2 * f12_inv

            # f12 gradient computation
            ∂f12 = @. -∂b_accum * (f1 * m[2:(end-2)] + f2 * m[3:(end-1)]) * f12_inv^2

            # f12 = f1 + f2 - accumulate gradients efficiently
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
    results = similar(tq, promote_type(eltype(u), eltype(tq)))

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
