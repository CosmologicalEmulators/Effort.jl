"""
    APWindowChebyshevPlan{T_mat, P, G, T_val}

A plan for combining Alcock-Paczynski (AP) effect and Window Function convolution 
using Chebyshev decomposition.

# Fields
- `M0`, `M2`, `M4`: Precomputed operator matrices (W * T) for monopole, quadrupole, and hexadecapole.
- `decomp_plan`: `ChebyshevPlan` for converting evaluations at nodes into coefficients.
- `sparse_k_nodes`: The k-nodes where the AP effect is evaluated (Chebyshev nodes).
- `k_min`, `k_max`: Domain of the Chebyshev expansion.
- `K`: Degree of the Chebyshev polynomial.
"""
struct APWindowChebyshevPlan{T_mat, P, G, T_val}
    M0::T_mat
    M2::T_mat
    M4::T_mat
    decomp_plan::P
    sparse_k_nodes::G
    k_min::T_val
    k_max::T_val
    K::Int
end

"""
    prepare_ap_window_chebyshev(W0, W2, W4, k_dense, k_min, k_max, K)

Precomputes the operators and plan for unified AP and window convolution.

# Arguments
- `W0`, `W2`, `W4`: Blocks of the window matrix corresponding to 0, 2, 4 multipoles.
- `k_dense`: The dense grid where the window matrix is defined.
- `k_min`, `k_max`: Domain for the Chebyshev expansion.
- `K`: Degree of the Chebyshev polynomial.
"""
function prepare_ap_window_chebyshev(W0::AbstractMatrix, W2::AbstractMatrix, W4::AbstractMatrix, 
                                    k_dense::AbstractVector, k_min::Real, k_max::Real, K::Int)
    # 1. Compute Chebyshev basis matrix T on the dense grid
    T_mat = chebyshev_polynomials(k_dense, k_min, k_max, K)
    
    # 2. Form the compressed operators M = W * T
    M0 = W0 * T_mat
    M2 = W2 * T_mat
    M4 = W4 * T_mat
    
    # 3. Create the decomposition plan and identify nodes
    decomp_plan = prepare_chebyshev_plan(k_min, k_max, K)
    sparse_k_nodes = decomp_plan.nodes[1] # Extract the vector of nodes
    
    return APWindowChebyshevPlan(M0, M2, M4, decomp_plan, sparse_k_nodes, k_min, k_max, K)
end

"""
    apply_AP_and_window(plan::APWindowChebyshevPlan, k_input, mono_in, quad_in, hexa_in, q_par, q_perp; 
                        n_GL_points=8, method=Cubic())

Combined application of Alcock-Paczynski effect and Window Function convolution.

Evaluates the AP-distorted power spectrum only at the Chebyshev nodes, decomposes 
it into coefficients, and applies the precomputed window operator.
"""
function apply_AP_and_window(plan::APWindowChebyshevPlan, k_input, mono_in, quad_in, hexa_in, q_par, q_perp; 
                            n_GL_points=8, method=Cubic())
    # Step 1: Evaluate AP on the sparse Chebyshev nodes
    # This returns (mono_AP, quad_AP, hexa_AP) evaluated at plan.sparse_k_nodes
    mono_AP, quad_AP, hexa_AP = apply_AP(k_input, plan.sparse_k_nodes, mono_in, quad_in, hexa_in, q_par, q_perp; 
                                       n_GL_points=n_GL_points, method=method)
                                       
    # Step 2: Decompose values into Chebyshev coefficients
    c0 = chebyshev_decomposition(plan.decomp_plan, mono_AP)
    c2 = chebyshev_decomposition(plan.decomp_plan, quad_AP)
    c4 = chebyshev_decomposition(plan.decomp_plan, hexa_AP)
    
    # Step 3: Apply the precomputed window operators
    # result_ell = sum_j M_{ell, j} * c_j
    # Note: If windows are block-diagonal, it's M0*c0, etc. 
    # If there is crossing (like in some EFT codes), M would be a larger block matrix.
    # Here we assume standard separate multipole windowing.
    p0_conv = plan.M0 * c0
    p2_conv = plan.M2 * c2
    p4_conv = plan.M4 * c4
    
    return p0_conv, p2_conv, p4_conv
end
