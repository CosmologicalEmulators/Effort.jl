@non_differentiable LinRange(a,b,n)
@non_differentiable _transformed_weights(quadrature_rule, order, a,b)

Zygote.@adjoint function _create_d(u, t, s, typed_zero)
    y = _create_d(u, t, s, typed_zero)
    function _create_d_pullback(ȳ)
        ∂u = Tridiagonal(zeros(eltype(typed_zero), s-1),
               map(i -> i == 1 ? typed_zero : 2 / (t[i] - t[i - 1]), 1:s),
               map(i -> - 2 / (t[i+1] - t[i]), 1:s-1)) * ȳ
        ∂t = Tridiagonal(zeros(eltype(typed_zero), s-1),
               map(i -> i == 1 ? typed_zero : -2 * (u[i] - u[i - 1]) / (t[i] - t[i - 1]) ^ 2, 1:s),
               map(i -> 2 * (u[i+1] - u[i]) / (t[i+1] - t[i]) ^ 2, 1:s-1)) * ȳ
        return (∂u, ∂t, NoTangent(), NoTangent())
    end
    return y, _create_d_pullback
end

Zygote.@adjoint function _create_σ(z, x, i_list)
    y = _create_σ(z, x, i_list)
    function _create_σ_pullback(ȳ)
        s = length(z)
        s1 = length(i_list)

        runner = 0.5 ./ (x[i_list] - x[i_list .- 1])
        runner_bis = 2. .* (z[i_list] - z[i_list .- 1])

        ∂z = (sparse(i_list, 1:s1 ,runner, s, s1) -
              sparse(i_list .- 1, 1:s1 ,runner, s, s1)) * ȳ
        ∂x = (-sparse(i_list, 1:s1 ,runner_bis .* runner .^2, s, s1) +
               sparse(i_list .- 1, 1:s1 , runner_bis .* runner .^2, s, s1)) * ȳ
        return (∂z, ∂x, NoTangent())
    end
    return y, _create_σ_pullback
end

Zygote.@adjoint function _compose(z, t, new_t, Cᵢ_list, s_new, i_list, σ)
    y = _compose(z, t, new_t, Cᵢ_list, s_new, i_list, σ)
    function _compose_pullback(ȳ)
        s = length(z)
        s1 = length(i_list)

        ∂z = sparse(i_list .-1, 1:s1, [new_t[j] - t[i_list[j] - 1] for j in 1:s_new], s, s1) * ȳ
        ∂t = sparse(i_list .-1, 1:s1, map(j -> -z[i_list[j] - 1]  - 2σ[j] * (new_t[j] - t[i_list[j] - 1]), 1:s_new), s, s1) * ȳ
        ∂t1 = Diagonal([+z[i_list[j] - 1]  + 2σ[j] * (new_t[j] - t[i_list[j] - 1]) for j in 1:s1]) * ȳ
        ∂σ = Diagonal(map(i -> (new_t[i] - t[i_list[i] - 1])^2, 1:s_new)) * ȳ
        ∂Cᵢ_list = Diagonal(ones(s1)) * ȳ
        return (∂z, ∂t, ∂t1, ∂Cᵢ_list, NoTangent(), NoTangent(), ∂σ)
    end
    return y, _compose_pullback
end

Zygote.@adjoint function _create_Cᵢ_list(u, i_list)
    y = _create_Cᵢ_list(u, i_list)
    function _create_Cᵢ_list_pullback(ȳ)
        s = length(z)
        s1 = length(i_list)
        ∂Cᵢ_list = sparse(i_list .-1, 1:s1 ,ones(s1), s, s1) * ȳ
        return (∂Cᵢ_list, NoTangent())
    end
    return y, _create_Cᵢ_list_pullback
end

Zygote.@adjoint function _create_i_list(t, new_t, s_new)
    y = _create_i_list(t, new_t, s_new)
    function _create_i_list_pullback(ȳ)
        return (NoTangent(), NoTangent(), NoTangent())
    end
    return y, _create_i_list_pullback
end
