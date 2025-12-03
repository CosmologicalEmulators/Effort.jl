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

