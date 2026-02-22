"""
    ChebyshevOperator{T_prime, P}

A linear operator M compressed using Chebyshev decomposition.

# Fields
- `M_prime::T_prime`: The transformed operator matrix M * T, where T is the Chebyshev basis matrix.
- `plan::P`: The `ChebyshevPlan` used for decomposition.
"""
struct ChebyshevOperator{T_prime, P}
    M_prime::T_prime
    plan::P
end

"""
    prepare_chebyshev_operator(M::AbstractMatrix, x_grid::AbstractVector, x_min::Real, x_max::Real, K::Int)

Precomputes a `ChebyshevOperator` by projecting the matrix M onto the Chebyshev basis.

# Arguments
- `M`: The original linear operator matrix (e.g., a window matrix).
- `x_grid`: The grid on which the input vector v is originally defined.
- `x_min`, `x_max`: The domain for the Chebyshev expansion.
- `K`: The degree of the Chebyshev polynomial.

# Returns
A `ChebyshevOperator` instance.
"""
function prepare_chebyshev_operator(M::AbstractMatrix{T}, x_grid::AbstractVector{T}, x_min::T, x_max::T, K::Int) where T
    # 1. Compute the Chebyshev basis matrix T on the original grid
    T_mat = chebyshev_polynomials(x_grid, x_min, x_max, K)
    
    # 2. Precompute the transformed operator M' = M * T
    M_prime = M * T_mat
    
    # 3. Create the Chebyshev plan for decomposition at runtime
    plan = prepare_chebyshev_plan(x_min, x_max, K)
    
    return ChebyshevOperator(M_prime, plan)
end

"""
    apply_chebyshev_operator(op::ChebyshevOperator, v_nodes::AbstractVector)

Applies the compressed operator to a function evaluated at the Chebyshev nodes.

# Arguments
- `op`: The `ChebyshevOperator`.
- `v_nodes`: The function values evaluated at `op.plan.nodes`.

# Returns
The result of M * v, computed as M' * c.
"""
function apply_chebyshev_operator(op::ChebyshevOperator, v_nodes::AbstractVector)
    # 1. Decompose function values into Chebyshev coefficients
    c = chebyshev_decomposition(op.plan, v_nodes)
    
    # 2. Apply the transformed operator
    return op.M_prime * c
end

"""
    apply_chebyshev_operator(op::ChebyshevOperator, v_nodes::AbstractMatrix)

Batch version of `apply_chebyshev_operator`.
"""
function apply_chebyshev_operator(op::ChebyshevOperator, v_nodes::AbstractMatrix)
    # 1. Decompose function values into Chebyshev coefficients (batch)
    c = chebyshev_decomposition(op.plan, v_nodes)
    
    # 2. Apply the transformed operator
    return op.M_prime * c
end
