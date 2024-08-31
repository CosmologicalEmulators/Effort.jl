function ChainRulesCore.rrule(::typeof(_simpson_weights), n)
    Y = _simpson_weights(n)
    function inv_maximin_pullback(Ȳ)
        return NoTangent(), NoTangent()
    end
    return Y, inv_maximin_pullback
end

@non_differentiable LinRange(a,b,n)
@non_differentiable _transformed_weights(quadrature_rule, order, a,b)
