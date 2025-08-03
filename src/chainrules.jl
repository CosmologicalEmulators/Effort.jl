@non_differentiable LinRange(a, b, n)
@non_differentiable _transformed_weights(quadrature_rule, order, a, b)
@non_differentiable gausslobatto(n)

@adjoint function window_convolution(W, v)
    C = window_convolution(W, v)
    function window_convolution_pullback(C̄)
        ∂W = @thunk(first_rule(C̄, v))
        ∂v = @thunk(second_rule(C̄, W))
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
        gm = copy(Δm)                # running adjoint of m

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

        # Initialize gradients
        ∂t = zeros(eltype(t), length(t))
        ∂m = zeros(eltype(m), length(m))

        # Pullback through d computation
        if Δd !== nothing
            # d = (b[1:(end - 1)] .+ b[2:end] .- 2 .* m[3:(end - 2)]) ./ dt.^2
            ∂b_from_d = zeros(eltype(b), length(b))
            ∂b_from_d[1:(end-1)] .+= Δd ./ dt .^ 2
            ∂b_from_d[2:end] .+= Δd ./ dt .^ 2
            ∂m[3:(end-2)] .-= 2 .* Δd ./ dt .^ 2

            ∂dt_from_d = -2 .* Δd .* (b[1:(end-1)] .+ b[2:end] .- 2 .* m[3:(end-2)]) ./ dt .^ 3
            ∂t[1:(end-1)] .-= ∂dt_from_d
            ∂t[2:end] .+= ∂dt_from_d

            Δb = Δb === nothing ? ∂b_from_d : Δb .+ ∂b_from_d
        end

        # Pullback through c computation
        if Δc !== nothing
            # c = (3 .* m[3:(end - 2)] .- 2 .* b[1:(end - 1)] .- b[2:end]) ./ dt
            ∂m[3:(end-2)] .+= 3 .* Δc ./ dt

            ∂b_from_c = zeros(eltype(b), length(b))
            ∂b_from_c[1:(end-1)] .-= 2 .* Δc ./ dt
            ∂b_from_c[2:end] .-= Δc ./ dt

            ∂dt_from_c = -Δc .* (3 .* m[3:(end-2)] .- 2 .* b[1:(end-1)] .- b[2:end]) ./ dt .^ 2
            ∂t[1:(end-1)] .-= ∂dt_from_c
            ∂t[2:end] .+= ∂dt_from_c

            Δb = Δb === nothing ? ∂b_from_c : Δb .+ ∂b_from_c
        end

        # Pullback through b computation
        if Δb !== nothing
            # b = (f1 .* m[2:(end-2)] .+ f2 .* m[3:(end-1)]) ./ f12
            ∂f1 = Δb .* m[2:(end-2)] ./ f12
            ∂f2 = Δb .* m[3:(end-1)] ./ f12
            ∂m[2:(end-2)] .+= Δb .* f1 ./ f12
            ∂m[3:(end-1)] .+= Δb .* f2 ./ f12
            ∂f12 = -Δb .* (f1 .* m[2:(end-2)] .+ f2 .* m[3:(end-1)]) ./ f12 .^ 2

            # f12 = f1 + f2
            ∂f1 .+= ∂f12
            ∂f2 .+= ∂f12

            # f1 = dm[3:(n + 2)], f2 = dm[1:n]
            ∂dm = zeros(eltype(dm), length(dm))
            ∂dm[3:(n+2)] .+= ∂f1
            ∂dm[1:n] .+= ∂f2

            # dm = abs.(diff(m))
            diff_m = diff(m)
            ∂diff_m = ∂dm .* sign.(diff_m)

            # diff(m) pullback
            ∂m[1:(end-1)] .-= ∂diff_m
            ∂m[2:end] .+= ∂diff_m
        end

        return (∂t, ∂m)
    end

    return (b, c, d), _akima_coefficients_pullback
end

@adjoint function _akima_eval(u, t, b, c, d, tq::AbstractArray)
    # Forward pass
    results = map(tqi -> _akima_eval(u, t, b, c, d, tqi), tq)

    function pullback(ȳ)
        # Initialize accumulated gradients
        ū_total = zero(u)
        t̄_total = zero(t)
        b̄_total = zero(b)
        c̄_total = zero(c)
        d̄_total = zero(d)
        tq̄ = similar(tq)

        # Accumulate gradients from each query point
        for i in eachindex(tq)
            idx = _akima_find_interval(t, tq[i])
            wj = tq[i] - t[idx]

            # Compute ∂f/∂wj for this query point
            dwj = 3 * d[idx] * wj^2 + 2 * c[idx] * wj + b[idx]

            # Accumulate gradients for this query point
            ū_total[idx] += ȳ[i]
            t̄_total[idx] += -ȳ[i] * dwj
            tq̄[i] = ȳ[i] * dwj
            b̄_total[idx] += ȳ[i] * wj
            c̄_total[idx] += ȳ[i] * wj^2
            d̄_total[idx] += ȳ[i] * wj^3
        end

        return ū_total, t̄_total, b̄_total, c̄_total, d̄_total, tq̄
    end

    return results, pullback
end
