@non_differentiable LinRange(a, b, n)
@non_differentiable _transformed_weights(quadrature_rule, order, a, b)
@non_differentiable gausslobatto(n)

Zygote.@adjoint function _interval_slopes(u::AbstractVector, t::AbstractVector)
    s = _interval_slopes(u, t)     # forward
    function back(Δs)
        # Δs has length n-1
        n = length(u)
        # gradients wrt u
        Δu = similar(u, eltype(Δs))
        fill!(Δu, 0)
        @inbounds for i in 1:n-1
            h = t[i+1] - t[i]
            Δu[i] -= Δs[i] / h
            Δu[i+1] += Δs[i] / h
        end
        # gradients wrt t
        Δt = similar(t, eltype(Δs))
        fill!(Δt, 0)
        @inbounds for i in 1:n-1
            diffu = u[i+1] - u[i]
            h = t[i+1] - t[i]
            Δt[i] += Δs[i] * diffu / h^2
            Δt[i+1] -= Δs[i] * diffu / h^2
        end
        return (Δu, Δt)
    end
    return s, back
end

Zygote.@adjoint function _extend_slopes(s::AbstractVector)
    s_ext = _extend_slopes(s)
    function back(Δs_ext)
        # strip the ghosts
        Δs = Δs_ext[3:end-2]
        # the two mirrored terms are added twice to first/last positions
        Δs[1] += 2Δs_ext[1] + 2Δs_ext[2]
        Δs[2] -= Δs_ext[1] + Δs_ext[2]
        Δs[end-1] -= Δs_ext[end-1] + Δs_ext[end]
        Δs[end] += 2Δs_ext[end-1] + 2Δs_ext[end]
        return (Δs,)
    end
    return s_ext, back
end

Zygote.@adjoint function _node_derivatives_makima(s_ext::AbstractVector)
    # 1. Run the forward pass using the (corrected) mutating function.
    d = _node_derivatives_makima(s_ext)

    # 2. Define the backward pass (the pullback).
    function node_derivatives_makima_pullback(Δd_in)
        # Ensure the incoming gradient is a concrete array
        Δd = unthunk(Δd_in)

        n = length(d)
        Δs_ext = zero(s_ext) # Initialize the output gradient

        # Loop over each calculated derivative to backpropagate its gradient
        for i in 1:n
            Δdi = Δd[i]
            if Δdi === nothing || iszero(Δdi)
                continue
            end

            # --- Recompute intermediates from the forward pass for this `i` ---
            s_ip1 = s_ext[i+1]
            s_ip2 = s_ext[i+2]
            s_ip3 = s_ext[i+3]
            s_i = s_ext[i]

            w1 = abs(s_ip3 - s_ip2) + abs(s_ip2 - s_ip1)
            w2 = abs(s_ip1 - s_i) + (i > 1 ? abs(s_i - s_ext[i-1]) : 0.0)
            w_sum = w1 + w2

            # --- Backpropagate based on the logic in the forward pass ---
            if w_sum == 0.0
                # Forward: d_i = (s_ip2 + s_ip1) / 2
                Δs_ext[i+2] += Δdi / 2.0
                Δs_ext[i+1] += Δdi / 2.0
            else
                # Forward: d_i = (w1*s_ip1 + w2*s_ip2) / w_sum
                # Apply chain rule for division: d(f/g) = (g*df - f*dg)/g^2
                d_i = d[i] # Value from forward pass

                ΔN = Δdi / w_sum
                ΔD = -Δdi * d_i / w_sum

                # From D = w1 + w2
                Δw1 = ΔD
                Δw2 = ΔD

                # From N = w1*s_ip1 + w2*s_ip2
                Δw1 += ΔN * s_ip1
                Δs_ext[i+1] += ΔN * w1
                Δw2 += ΔN * s_ip2
                Δs_ext[i+2] += ΔN * w2

                # Backpropagate through w1 = abs(s_ip3-s_ip2) + abs(s_ip2-s_ip1)
                sign_1a = sign(s_ip3 - s_ip2)
                sign_1b = sign(s_ip2 - s_ip1)
                Δs_ext[i+3] += Δw1 * sign_1a
                Δs_ext[i+2] += Δw1 * (-sign_1a + sign_1b)
                Δs_ext[i+1] += Δw1 * (-sign_1b)

                # Backpropagate through w2, respecting the boundary condition
                sign_2a = sign(s_ip1 - s_i)
                Δs_ext[i+1] += Δw2 * sign_2a
                if i > 1
                    sign_2b = sign(s_i - s_ext[i-1])
                    Δs_ext[i] += Δw2 * (-sign_2a + sign_2b)
                    Δs_ext[i-1] += Δw2 * (-sign_2b)
                else
                    # For i=1, the `abs(s_i - s_ext[i-1])` term was zero, so no gradient flows
                    # back from it. The only contribution is from `abs(s_ip1 - s_i)`.
                    Δs_ext[i] += Δw2 * (-sign_2a)
                end
            end
        end

        # Return gradient as a tuple, one for each input argument.
        return (Δs_ext,)
    end

    return d, node_derivatives_makima_pullback
end

Zygote.@adjoint function _get_i_list(t, x)
    y = _get_i_list(t, x)
    function get_i_list_pullback(ȳ)
        return (NoTangent(), NoTangent())
    end
    return y, get_i_list_pullback
end

Zygote.@adjoint function _akima_eval(u, t, d, x)
    y = _akima_eval(u, t, d, x)
    function pullback(ȳ)
        m = length(x)
        ilist = get_i_list(t, x)
        ∇u = zeros(eltype(u), length(u))
        ∇d = zeros(eltype(d), length(d))
        ∇t = zeros(eltype(t), length(t))
        ∇x = zeros(eltype(x), length(x))

        for i in 1:m
            j = ilist[i]
            t0 = t[j]
            t1 = t[j+1]
            h = t1 - t0
            xi = x[i]
            ξ = (xi - t0) / h

            # Hermite basis and their derivatives
            ξ2 = ξ^2
            ξ3 = ξ^3
            h00 = 2 * ξ3 - 3 * ξ2 + 1
            h10 = ξ3 - 2 * ξ2 + ξ
            h01 = -2 * ξ3 + 3 * ξ2
            h11 = ξ3 - ξ2
            dh00 = 6 * ξ2 - 6 * ξ
            dh10 = 3 * ξ2 - 4 * ξ + 1
            dh01 = -6 * ξ2 + 6 * ξ
            dh11 = 3 * ξ2 - 2 * ξ

            # Output w.r.t. inputs
            u0 = u[j]
            u1 = u[j+1]
            d0 = d[j]
            d1 = d[j+1]

            # Gradients w.r.t. u and d
            ∇u[j] += h00 * ȳ[i]
            ∇u[j+1] += h01 * ȳ[i]
            ∇d[j] += h10 * h * ȳ[i]
            ∇d[j+1] += h11 * h * ȳ[i]

            # Derivatives for t and x
            # ∂y/∂ξ
            dy_dξ = dh00 * u0 + dh10 * h * d0 + dh01 * u1 + dh11 * h * d1
            # ∂y/∂h (explicitly, holding ξ constant)
            dy_dh = h10 * d0 + h11 * d1

            # ---- CORRECTED SECTION ----
            # Corrected partial derivatives of ξ w.r.t. t
            dξ_dt0 = (ξ - 1) / h
            dξ_dt1 = -ξ / h
            # -------------------------

            # ∂h/∂t0, ∂h/∂t1
            dh_dt0 = -1
            dh_dt1 = +1

            # Chain rule for t[j]
            ∇t[j] += (dy_dξ * dξ_dt0 + dy_dh * dh_dt0) * ȳ[i]
            # Chain rule for t[j+1]
            ∇t[j+1] += (dy_dξ * dξ_dt1 + dy_dh * dh_dt1) * ȳ[i]

            # Chain rule for x[i]
            dξ_dx = 1 / h
            ∇x[i] += dy_dξ * dξ_dx * ȳ[i]
        end

        return (∇u, ∇t, ∇d, ∇x)
    end
    return y, pullback
end

Zygote.@adjoint function window_convolution(W, v)
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
