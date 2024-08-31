function _transformed_weights(quadrature_rule, order, a,b)
    x, w = quadrature_rule(order)
    x = (b-a)/2. .* x .+ (b+a)/2.
    w = (b-a)/2. .* w
    return x, w
end
