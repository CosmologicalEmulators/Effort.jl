"""
    _Pkμ(k, μ, Int_Mono, Int_Quad, Int_Hexa)

Reconstructs the anisotropic power spectrum `` P(k, \\mu) `` at a given wavenumber `k`
and cosine of the angle to the line-of-sight `μ`, using its Legendre multipole moments.

# Arguments
- `k`: The wavenumber.
- `μ`: The cosine of the angle to the line-of-sight.
- `Int_Mono`: A function or interpolant that provides the monopole moment `` I_0(k) `` at wavenumber `k`.
- `Int_Quad`: A function or interpolant that provides the quadrupole moment `` I_2(k) `` at wavenumber `k`.
- `Int_Hexa`: A function or interpolant that provides the hexadecapole moment `` I_4(k) `` at wavenumber `k`.

# Returns
The value of the anisotropic power spectrum `` P(k, \\mu) `` at the given `k` and `μ`.

# Details
The anisotropic power spectrum is reconstructed as a sum of its multipole moments
multiplied by the corresponding Legendre polynomials evaluated at `μ`. The function
uses the 0th, 2nd, and 4th order Legendre polynomials.

# Formula
The formula used is:
```math
P(k, \\mu) = I_0(k) \\mathcal{L}_0(\\mu) + I_2(k) \\mathcal{L}_2(\\mu) + I_4(k) \\mathcal{L}_4(\\mu)
```
where `` I_l(k) `` are the multipole moments and `` \\mathcal{L}_l(\\mu) `` are the Legendre polynomials of order `` l ``.

# See Also
- [`_Legendre_0`](@ref): Calculates the 0th order Legendre polynomial.
- [`_Legendre_2`](@ref): Calculates the 2nd order Legendre polynomial.
- [`_Legendre_4`](@ref): Calculates the 4th order Legendre polynomial.
"""
function _Pkμ(k, μ, Int_Mono, Int_Quad, Int_Hexa)
    return Int_Mono(k) * _Legendre_0(μ) + Int_Quad(k) * _Legendre_2(μ) + Int_Hexa(k) * _Legendre_4(μ)
end

"""
    _k_true(k_o, μ_o, q_perp, F)

Calculates the true (physical) wavenumber `k` from the observed wavenumber `k_o`
and observed cosine of the angle to the line-of-sight `μ_o`.

This transformation accounts for anisotropic effects, likely redshift-space distortions (RSD)
or anisotropic cosmological scaling, parameterized by `q_perp` and `F`.

# Arguments
- `k_o`: The observed wavenumber (scalar).
- `μ_o`: The observed cosine of the angle to the line-of-sight (scalar).
- `q_perp`: A parameter related to perpendicular anisotropic scaling.
- `F`: A parameter related to parallel anisotropic scaling (often the growth rate `f` divided by the anisotropic scaling parameter `q_parallel`).

# Returns
The calculated true wavenumber `k` (scalar).

# Formula
The formula used is:
```math
k = \\frac{k_o}{q_\\perp} \\sqrt{1 + \\mu_o^2 \\left(\\frac{1}{F^2} - 1\\right)}
```

# See Also
- [`_k_true(k_o::Array, μ_o::Array, q_perp, F)`](@ref): Method for arrays of observed values.
- [`_μ_true`](@ref): Calculates the true cosine of the angle to the line-of-sight.
"""
function _k_true(k_o, μ_o, q_perp, F)
    return k_o / q_perp * sqrt(1 + μ_o^2 * (1 / F^2 - 1))
end

"""
    _k_true(k_o::Array, μ_o::Array, q_perp, F)

Calculates the true (physical) wavenumber `k` for arrays of observed wavenumbers `k_o`
and observed cosines of the angle to the line-of-sight `μ_o`.

This method applies the transformation from observed to true wavenumber element-wise
or for combinations of input arrays, accounting for anisotropic effects parameterized
by `q_perp` and `F`.

# Arguments
- `k_o`: An array of observed wavenumbers.
- `μ_o`: An array of observed cosines of the angle to the line-of-sight.
- `q_perp`: A parameter related to perpendicular anisotropic scaling.
- `F`: A parameter related to parallel anisotropic scaling.

# Returns
A vector containing the calculated true wavenumbers `k` for the given input arrays.

# Details
The function calculates `k` for pairs or combinations of values from the input arrays
`k_o` and `μ_o` using a formula derived from anisotropic scaling. The calculation involves
broadcasting and array operations to handle the array inputs efficiently. The result
is flattened into a vector.

# Formula
The underlying transformation for each pair of `k_o` and `μ_o` is:
```math
k = \\frac{k_o}{q_\\perp} \\sqrt{1 + \\mu_o^2 \\left(\\frac{1}{F^2} - 1\\right)}
```

# See Also
- [`_k_true(k_o, μ_o, q_perp, F)`](@ref): Method for scalar observed values.
- [`_μ_true`](@ref): Calculates the true cosine of the angle to the line-of-sight.
"""
function _k_true(k_o::Array, μ_o::Array, q_perp, F)
    a = @. sqrt(1 + μ_o^2 * (1 / F^2 - 1))
    result = (k_o ./ q_perp) * a'
    return vec(result)
end

"""
    _μ_true(μ_o, F)

Calculates the true (physical) cosine of the angle to the line-of-sight `μ` from the
observed cosine of the angle to the line-of-sight `μ_o`.

This transformation accounts for anisotropic effects, likely redshift-space distortions (RSD)
or anisotropic cosmological scaling, parameterized by `F`.

# Arguments
- `μ_o`: The observed cosine of the angle to the line-of-sight (scalar).
- `F`: A parameter related to parallel anisotropic scaling (often the growth rate `f` divided by the anisotropic scaling parameter `q_parallel`).

# Returns
The calculated true cosine of the angle to the line-of-sight `μ` (scalar).

# Formula
The formula used is:
```math
\\mu = \\frac{\\mu_o}{F \\sqrt{1 + \\mu_o^2 \\left(\\frac{1}{F^2} - 1\\right)}}
```

# See Also
- [`_μ_true(μ_o::Array, F)`](@ref): Method for an array of observed values.
- [`_k_true`](@ref): Calculates the true wavenumber.
"""
function _μ_true(μ_o, F)
    return μ_o / F / sqrt(1 + μ_o^2 * (1 / F^2 - 1))
end

"""
    _μ_true(μ_o::Array, F)

Calculates the true (physical) cosine of the angle to the line-of-sight `μ` for an array
of observed cosines of the angle to the line-of-sight `μ_o`.

This method applies the transformation from observed to true angle cosine element-wise,
accounting for anisotropic effects parameterized by `F`.

# Arguments
- `μ_o`: An array of observed cosines of the angle to the line-of-sight.
- `F`: A parameter related to parallel anisotropic scaling.

# Returns
An array containing the calculated true cosines of the angle to the line-of-sight `μ`.

# Details
The function calculates `μ` for each value in the input array `μ_o` using a formula
derived from anisotropic scaling. Broadcasting (`@.`) is used to apply the calculation
element-wise.

# Formula
The underlying transformation for each `μ_o` is:
```math
\\mu = \\frac{\\mu_o}{F \\sqrt{1 + \\mu_o^2 \\left(\\frac{1}{F^2} - 1\\right)}}
```

# See Also
- [`_μ_true(μ_o, F)`](@ref): Method for a scalar observed value.
- [`_k_true`](@ref): Calculates the true wavenumber.
"""
function _μ_true(μ_o::Array, F)
    a = @. 1 / sqrt(1 + μ_o^2 * (1 / F^2 - 1))
    result = (μ_o ./ F) .* a
    return result
end

"""
    _P_obs(k_o, μ_o, q_par, q_perp, Int_Mono, Int_Quad, Int_Hexa)

Calculates the observed power spectrum `` P_{\\text{obs}}(k_o, \\mu_o) `` at a given observed
wavenumber `k_o` and observed cosine of the angle to the line-of-sight `μ_o`.

This function transforms the observed coordinates to true (physical) coordinates,
calculates the true power spectrum using provided interpolants for the multipole
moments, and applies the appropriate scaling factor due to anisotropic effects.

# Arguments
- `k_o`: The observed wavenumber.
- `μ_o`: The observed cosine of the angle to the line-of-sight.
- `q_par`: A parameter related to parallel anisotropic scaling.
- `q_perp`: A parameter related to perpendicular anisotropic scaling.
- `Int_Mono`: An interpolation function for the monopole moment `` I_0(k) `` in true k.
- `Int_Quad`: An interpolation function for the quadrupole moment `` I_2(k) `` in true k.
- `Int_Hexa`: An interpolation function for the hexadecapole moment `` I_4(k) `` in true k.

# Returns
The value of the observed power spectrum `` P_{\\text{obs}}(k_o, \\mu_o) ``.

# Details
The observed coordinates `` (k_o, \\mu_o) `` are transformed to true coordinates `` (k_t, \\mu_t) ``
using the [`_k_true`](@ref) and [`_μ_true`](@ref) functions, with `` F = q_\\parallel / q_\\perp ``.
The true power spectrum `` P(k_t, \\mu_t) `` is then reconstructed using [`_Pkμ`](@ref)
and the provided multipole interpolants. Finally, the result is scaled by `` 1 / (q_\\parallel q_\\perp^2) ``.

# Formula
The formula used is:
```math
P_{\\text{obs}}(k_o, \\mu_o) = \\frac{1}{q_\\parallel q_\\perp^2} P(k_t, \\mu_t)
```
where
```math
k_t = \\text{_k_true}(k_o, \\mu_o, q_\\perp, F)
```
```math
\\mu_t = \\text{_μ_true}(\\mu_o, F)
```
and
```math
F = q_\\parallel / q_\\perp
```

# See Also
- [`_k_true`](@ref): Transforms observed wavenumber to true wavenumber.
- [`_μ_true`](@ref): Transforms observed angle cosine to true angle cosine.
- [`_Pkμ`](@ref): Reconstructs the true power spectrum from multipole moments.
- [`interp_Pℓs`](@ref): Creates the multipole interpolants.
"""
function _P_obs(k_o, μ_o, q_par, q_perp, Int_Mono, Int_Quad, Int_Hexa)
    F = q_par / q_perp
    k_t = _k_true(k_o, μ_o, q_perp, F)
    μ_t = _μ_true(μ_o, F)

    return _Pkμ(k_t, μ_t, Int_Mono, Int_Quad, Int_Hexa) / (q_par * q_perp^2)
end

"""
    interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)

Creates interpolation functions for the monopole, quadrupole, and hexadecapole
moments of the power spectrum.

These interpolants can then be used to efficiently evaluate the multipole moments
at arbitrary wavenumbers `k`.

# Arguments
- `Mono_array`: An array containing the values of the monopole moment `` I_0(k) ``.
- `Quad_array`: An array containing the values of the quadrupole moment `` I_2(k) ``.
- `Hexa_array`: An array containing the values of the hexadecapole moment `` I_4(k) ``.
- `k_grid`: An array containing the corresponding wavenumber `k` values for the multipole arrays.

# Returns
A tuple containing three interpolation functions: `(Int_Mono, Int_Quad, Int_Hexa)`.

# Details
The function uses `AkimaInterpolation` from the `Interpolations.jl` package to create
the interpolants. Extrapolation is set to `ExtrapolationType.Extension`, which means
the interpolant will use the nearest data points to extrapolate outside the provided
`k_grid` range. Note that extrapolation can sometimes introduce errors.

# See Also
- [`_Pkμ`](@ref): Uses the interpolation functions to reconstruct the anisotropic power spectrum.
"""
function interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_grid)
    #extrapolation might introduce some errors ar high k, when q << 1.
    #maybe we should implement a log extrapolation?
    Int_Mono = AkimaInterpolation(Mono_array, k_grid; extrapolation=ExtrapolationType.Extension)
    Int_Quad = AkimaInterpolation(Quad_array, k_grid; extrapolation=ExtrapolationType.Extension)
    Int_Hexa = AkimaInterpolation(Hexa_array, k_grid; extrapolation=ExtrapolationType.Extension)
    return Int_Mono, Int_Quad, Int_Hexa
end

"""
    apply_AP_check(k_input::Array, k_output::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par, q_perp)

Calculates the observed power spectrum multipole moments (monopole, quadrupole, hexadecapole)
on a given observed wavenumber grid `k_output`, from arrays of true multipole moments
provided on an input wavenumber grid `k_input`, using numerical integration.

This is a **check version**, intended for verifying results from faster methods. It is
significantly slower due to the use of numerical integration over the angle `μ`.

# Arguments
- `k_input`: An array of wavenumber values on which the input true multipole moments (`Mono_array`, `Quad_array`, `Hexa_array`) are defined.
- `k_output`: An array of observed wavenumber values at which to calculate the output observed multipoles.
- `Mono_array`: An array containing the values of the true monopole moment `` I_0(k) `` on the `k_input` grid.
- `Quad_array`: An array containing the values of the true quadrupole moment `` I_2(k) `` on the `k_input` grid.
- `Hexa_array`: An array containing the values of the true hexadecapole moment `` I_4(k) `` on the `k_input` grid.
- `q_par`: A parameter related to parallel anisotropic scaling.
- `q_perp`: A parameter related to perpendicular anisotropic scaling.

# Returns
A tuple `(P0_obs, P2_obs, P4_obs)`, where each element is an array containing the calculated
observed monopole, quadrupole, and hexadecapole moments respectively, evaluated at the
wavenumbers in `k_output`.

# Details
This method first creates interpolation functions for the true multipole moments using
[`interp_Pℓs`](@ref) based on the `k_input` grid. It then calls the core [`apply_AP_check(k_grid, int_Mono, int_Quad, int_Hexa, q_par, q_perp)`](@ref)
method, passing `k_output` as the grid at which to calculate the observed multipoles.

This function is a **slower check implementation** and should not be used in performance-critical code.

# Formula
The observed multipole moments are calculated using the formula:
```math
P_\\ell(k_o) = (2\\ell + 1) \\int_{0}^1 P_{\\text{obs}}(k_o, \\mu_o) \\mathcal{L}_\\ell(\\mu_o) d\\mu_o
```
for `` \\ell \\in \\{0, 2, 4\\} ``. The observed power spectrum `` P_{\\text{obs}}(k_o, \\mu_o) ``
is calculated using [`_P_obs(k_o, μ_o, q_par, q_perp, int_Mono, int_Quad, int_Hexa)`](@ref).

# See Also
- [`apply_AP_check(k_grid, int_Mono, int_Quad, int_Hexa, q_par, q_perp)`](@ref): The core method performing the integration.
- [`interp_Pℓs`](@ref): Creates the interpolation functions for the true multipoles.
- [`_P_obs`](@ref): Calculates the observed power spectrum.
"""
function apply_AP_check(k_input::Array, k_output::Array, Mono_array::Array, Quad_array::Array,
    Hexa_array::Array, q_par, q_perp)
    int_Mono, int_Quad, int_Hexa = interp_Pℓs(Mono_array, Quad_array, Hexa_array, k_input)
    return apply_AP_check(k_output, int_Mono, int_Quad, int_Hexa, q_par, q_perp)
end

function apply_AP_check(k_grid, int_Mono::DataInterpolations.AbstractInterpolation,
    int_Quad::DataInterpolations.AbstractInterpolation,
    int_Hexa::DataInterpolations.AbstractInterpolation, q_par, q_perp)
    nk = length(k_grid)
    result = zeros(3, nk)
    ℓ_array = [0, 2, 4]
    for i in 1:nk # TODO: use enumerate(k_grid)
        for (ℓ_idx, myℓ) in enumerate(ℓ_array)
            # Define the integrand function (Integrals.jl requires a parameter argument)
            integrand = (x, p) -> Pl(x, myℓ) * _P_obs(k_grid[i], x, q_par,
                    q_perp, int_Mono, int_Quad, int_Hexa)
            # Create and solve the integral problem using Integrals.jl
            prob = IntegralProblem(integrand, (0.0, 1.0))
            sol = solve(prob, QuadGKJL(), reltol=1e-12)
            result[ℓ_idx, i] = (2 * myℓ + 1) * sol.u
        end
    end
    return result[1, :], result[2, :], result[3, :]
end

"""
    q_par_perp(z, cosmo_mcmc::AbstractCosmology, cosmo_ref::AbstractCosmology)

Calculates the parallel (`q_par`) and perpendicular (`q_perp`) Alcock-Paczynski (AP)
parameters at a given redshift `z`, comparing a varying cosmology to a reference cosmology.

The AP parameters quantify the distortion of observed clustering due to assuming a
different cosmology than the true one when converting redshifts and angles to distances.

# Arguments
- `z`: The redshift at which to calculate the AP parameters.
- `cosmo_mcmc`: An `AbstractCosmology` struct representing the varying cosmology (e.g., from an MCMC chain).
- `cosmo_ref`: An `AbstractCosmology` struct representing the reference cosmology used for measurements.

# Returns
A tuple `(q_par, q_perp)` containing the calculated parallel and perpendicular AP parameters at redshift `z`.

# Details
The parallel AP parameter `q_par` is the ratio of the Hubble parameter in the reference
cosmology to that in the varying cosmology. The perpendicular AP parameter `q_perp` is
the ratio of the conformal angular diameter distance in the varying cosmology to that
in the reference cosmology.

# Formula
The formulas for the Alcock-Paczynski parameters are:
```math
q_\\parallel(z) = \\frac{E_{\\text{ref}}(z)}{E_{\\text{mcmc}}(z)}
```
```math
q_\\perp(z) = \\frac{\\tilde{d}_{A,\\text{mcmc}}(z)}{\\tilde{d}_{A,\\text{ref}}(z)}
```
where `` E(z) `` is the normalized Hubble parameter and `` \\tilde{d}_A(z) `` is the conformal
angular diameter distance.
"""
function q_par_perp(z, cosmo_mcmc::AbstractCosmology, cosmo_ref::AbstractCosmology)
    E_ref = E_z(z, cosmo_ref)
    E_mcmc = E_z(z, cosmo_mcmc)

    d̃A_ref = d̃A_z(z, cosmo_ref)
    d̃A_mcmc = d̃A_z(z, cosmo_mcmc)

    q_perp = d̃A_mcmc / d̃A_ref
    q_par = E_ref / E_mcmc

    return q_par, q_perp
end

"""
    _Pk_recon(mono::Matrix, quad::Matrix, hexa::Matrix, l0, l2, l4)

Reconstructs the anisotropic power spectrum `` P(k, \\mu) `` on a grid of wavenumbers `k`
and cosines of the angle to the line-of-sight `μ`, using matrices of its Legendre
multipole moments and vectors of Legendre polynomial values.

This function is designed to efficiently reconstruct the 2D power spectrum for multiple
`k` and `μ` values simultaneously, assuming the multipole moments are provided as
matrices (e.g., `N_k x 1`) and Legendre polynomials as vectors (e.g., `N_μ`).

# Arguments
- `mono`: A matrix containing the monopole moment `` I_0(k) `` values (expected dimensions `N_k x 1`).
- `quad`: A matrix containing the quadrupole moment `` I_2(k) `` values (expected dimensions `N_k x 1`).
- `hexa`: A matrix containing the hexadecapole moment `` I_4(k) `` values (expected dimensions `N_k x 1`).
- `l0`: A vector containing the 0th order Legendre polynomial `` \\mathcal{L}_0(\\mu) `` values evaluated at the desired `μ` values (expected dimensions `N_μ`).
- `l2`: A vector containing the 2nd order Legendre polynomial `` \\mathcal{L}_2(\\mu) `` values evaluated at the desired `μ` values (expected dimensions `N_μ`).
- `l4`: A vector containing the 4th order Legendre polynomial `` \\mathcal{L}_4(\\mu) `` values evaluated at the desired `μ` values (expected dimensions `N_μ`).

# Returns
A matrix representing the anisotropic power spectrum `` P(k, \\mu) `` on the `N_k x N_μ` grid.

# Details
The function reconstructs the anisotropic power spectrum using the formula that sums
the multipole moments multiplied by the corresponding Legendre polynomials. The matrix
and vector operations are broadcast to calculate the result for all combinations of
input `k` (from the rows of the moment matrices) and `μ` (from the elements of the
Legendre polynomial vectors).

# Formula
The formula used for each element `` (i, j) `` of the output matrix (corresponding to the `` i ``-th
wavenumber and `` j ``-th angle cosine) is:
```math
P(k_i, \\mu_j) = I_0(k_i) \\mathcal{L}_0(\\mu_j) + I_2(k_i) \\mathcal{L}_2(\\mu_j) + I_4(k_i) \\mathcal{L}_4(\\mu_j)
```

# See Also
- [`_Pkμ`](@ref): Reconstructs `` P(k, \\mu) `` for single `k` and `μ`.
- [`_Legendre_0`](@ref), [`_Legendre_2`](@ref), [`_Legendre_4`](@ref): Calculate the Legendre polynomials.
"""
function _Pk_recon(mono::Matrix, quad::Matrix, hexa::Matrix, l0, l2, l4)
    return mono .* l0' .+ quad .* l2' + hexa .* l4'
end

"""
    apply_AP(k_input::Array, k_output::Array, mono::Array, quad::Array, hexa::Array, q_par, q_perp; n_GL_points=8)

Calculates the observed power spectrum multipole moments (monopole, quadrupole, hexadecapole)
on a given observed wavenumber grid `k_output`, using arrays of true multipole moments
provided on an input wavenumber grid `k_input`, and employing Gauss-Lobatto quadrature.

This is the **standard, faster implementation** for applying the Alcock-Paczynski (AP)
effect to the power spectrum multipoles, designed
for performance compared to the check version using generic numerical integration.

# Arguments
- `k_input`: An array of wavenumber values on which the input true multipole moments (`mono`, `quad`, `hexa`) are defined.
- `k_output`: An array of observed wavenumber values at which to calculate the output observed multipoles.
- `mono`: An array containing the values of the true monopole moment `` I_0(k) `` on the `k_input` grid.
- `quad`: An array containing the values of the true quadrupole moment `` I_2(k) `` on the `k_input` grid.
- `hexa`: An array containing the values of the true hexadecapole moment `` I_4(k) `` on the `k_input` grid.
- `q_par`: A parameter related to parallel anisotropic scaling.
- `q_perp`: A parameter related to perpendicular anisotropic scaling.

# Keyword Arguments
- `n_GL_points`: The number of Gauss-Lobatto points to use for the integration over `μ`. The actual number of nodes used corresponds to `2 * n_GL_points`. Defaults to 8.

# Returns
A tuple `(P0_obs, P2_obs, P4_obs)`, where each element is an array containing the calculated
observed monopole, quadrupole, and hexadecapole moments respectively, evaluated at the
observed wavenumbers in `k_output`.

# Details
The function applies the AP and RSD effects by integrating the observed anisotropic
power spectrum `` P_{\\text{obs}}(k_o, \\mu_o) `` over the observed cosine of the angle
to the line-of-sight `` \\mu_o \\in [0, 1] `` (assuming symmetry for even multipoles),
weighted by the corresponding Legendre polynomial `` \\mathcal{L}_\\ell(\\mu_o) ``.

The process involves:
1. Determine Gauss-Lobatto nodes and weights for the interval `[0, 1]`.
2. For each observed wavenumber `k_o` in the input `k_output` array and each `μ_o` node:
   a. Calculate the true wavenumber `` k_t(k_o, \\mu_o) `` using [`_k_true`](@ref).
   b. Calculate the true angle cosine `` \\mu_t(\\mu_o) `` using [`_μ_true`](@ref).
   c. Interpolate the true multipole moments `` I_\\ell(k_t) `` using [`_akima_spline_legacy`](@ref), interpolating from the `k_input` grid to the new `k_t` values.
   d. Calculate the true Legendre polynomials `` \\mathcal{L}_\\ell(\\mu_t) `` using [`_Legendre_0`](@ref), [`_Legendre_2`](@ref), [`_Legendre_4`](@ref).
   e. Reconstruct the true power spectrum `` P(k_t, \\mu_t) `` using [`_Pk_recon`](@ref).
   f. Calculate the observed power spectrum `` P_{\\text{obs}}(k_o, \\mu_o) = P(k_t, \\mu_t) / (q_\\parallel q_\\perp^2) ``.
3. Perform the weighted sum (quadrature) over the `μ_o` nodes to get the observed multipoles `` P_\\ell(k_o) `` on the `k_output` grid.

This function is the **standard, performant implementation** for applying AP compared to the slower [`apply_AP_check`](@ref).

# Formula
The observed multipole moments are calculated using the formula:
```math
P_\\ell(k_o) = (2\\ell + 1) \\int_{0}^1 P_{\\text{obs}}(k_o, \\mu_o) \\mathcal{L}_\\ell(\\mu_o) d\\mu_o
```
for `` \\ell \\in \\{0, 2, 4\\} ``. The integral is approximated using Gauss-Lobatto quadrature.

# See Also
- [`apply_AP_check`](@ref): The slower, check version using generic numerical integration.
- [`_k_true`](@ref): Transforms observed wavenumber to true wavenumber.
- [`_μ_true`](@ref): Transforms observed angle cosine to true angle cosine.
- [`_Legendre_0`](@ref), [`_Legendre_2`](@ref), [`_Legendre_4`](@ref): Calculate the Legendre polynomials.
- [`_akima_spline`](@ref): Interpolates the true multipole moments.
- [`_Pk_recon`](@ref): Reconstructs the true power spectrum on a grid.
- `gausslobatto`: Function used to get quadrature nodes and weights.
"""
function apply_AP(k_input::Array, k_output::Array, mono::Array, quad::Array, hexa::Array, q_par, q_perp;
    n_GL_points=8)
    nk = length(k_output)
    nodes, weights = gausslobatto(n_GL_points * 2)
    #since the integrand is symmetric, we are gonna use only half of the points
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    F = q_par / q_perp

    k_t = _k_true(k_output, μ_nodes, q_perp, F)

    μ_t = _μ_true(μ_nodes, F)

    Pl0_t = _Legendre_0.(μ_t)
    Pl2_t = _Legendre_2.(μ_t)
    Pl4_t = _Legendre_4.(μ_t)

    Pl0 = _Legendre_0.(μ_nodes) .* μ_weights .* (2 * 0 + 1)
    Pl2 = _Legendre_2.(μ_nodes) .* μ_weights .* (2 * 2 + 1)
    Pl4 = _Legendre_4.(μ_nodes) .* μ_weights .* (2 * 4 + 1)

    new_mono = reshape(_akima_spline_legacy(mono, k_input, k_t), nk, n_GL_points)
    new_quad = reshape(_akima_spline_legacy(quad, k_input, k_t), nk, n_GL_points)
    new_hexa = reshape(_akima_spline_legacy(hexa, k_input, k_t), nk, n_GL_points)

    Pkμ = _Pk_recon(new_mono, new_quad, new_hexa, Pl0_t, Pl2_t, Pl4_t) ./ (q_par * q_perp^2)

    return Pkμ * Pl0, Pkμ * Pl2, Pkμ * Pl4
end

"""
    apply_AP(k_input::Array, k_output::Array, mono::Matrix, quad::Matrix, hexa::Matrix, q_par, q_perp; n_GL_points=8)

Batch version of `apply_AP` for processing multiple columns simultaneously.

This method applies the Alcock-Paczynski effect to multiple sets of multipole moments
(e.g., multiple Jacobian columns or parameter variations) in a single call.

# Arguments
- `k_input::Array`: Input wavenumber grid.
- `k_output::Array`: Output wavenumber grid.
- `mono::Matrix`: Monopole moments with shape `(n_k, n_cols)`.
- `quad::Matrix`: Quadrupole moments with shape `(n_k, n_cols)`.
- `hexa::Matrix`: Hexadecapole moments with shape `(n_k, n_cols)`.
- `q_par`: Parallel AP parameter.
- `q_perp`: Perpendicular AP parameter.

# Keyword Arguments
- `n_GL_points::Int`: Number of Gauss-Lobatto points. Default: 8.

# Returns
A tuple `(mono_AP, quad_AP, hexa_AP)` where each is a matrix of shape `(n_k_output, n_cols)`
containing the AP-corrected multipoles for all input columns.

# Details
This function iterates over each column of the input matrices, applies the AP effect
using the single-column `apply_AP` method, and stacks the results back into matrices.

This is particularly useful for computing Jacobians where each column represents the
derivative with respect to a different parameter.

# See Also
- [`apply_AP(k_input::Array, k_output::Array, mono::Array, quad::Array, hexa::Array, q_par, q_perp)`](@ref): Single-column version.
"""
function apply_AP(k_input::Array, k_output::Array, mono::Matrix, quad::Matrix, hexa::Matrix, q_par, q_perp;
    n_GL_points=8)

    results = [apply_AP(k_input, k_output, mono[:, i], quad[:, i], hexa[:, i],
        q_par, q_perp, n_GL_points=n_GL_points) for i in 1:size(mono, 2)]

    matrix1 = stack([tup[1] for tup in results], dims=2)
    matrix2 = stack([tup[2] for tup in results], dims=2)
    matrix3 = stack([tup[3] for tup in results], dims=2)

    return matrix1, matrix2, matrix3
end

"""
    window_convolution(W::Array{T, 4}, v::Matrix) where {T}

Applies a 4-dimensional window function or kernel `W` to a 2-dimensional input matrix `v`.

This operation performs a transformation or generalized convolution, summing over the
`j` and `l` indices of the inputs to produce a 2D result indexed by `i` and `k`.
This is commonly used in analyses where a 4D kernel relates input data in two dimensions
to output data in another two dimensions.

# Arguments
- `W`: A 4-dimensional array representing the window function or kernel.
- `v`: A 2-dimensional matrix representing the input data.

# Returns
A 2-dimensional matrix representing the result of the convolution or transformation.

# Details
The function implements the summation using the `@tullio` macro, which provides
an efficient way to express tensor contractions and generalized convolutions.
The operation can be thought of as applying a 4D kernel to a 2D input, resulting
in a 2D output.

# Formula
The operation is defined as:
```math
C_{ik} = \\sum_{j,l} W_{ijkl} v_{jl}
```

# See Also
- [`window_convolution(W::AbstractMatrix, v::AbstractVector)`](@ref): Method for a matrix kernel and vector input.

# References
- The methodology for this type of window measurement is discussed in: [arXiv:1810.05051](https://arxiv.org/abs/1810.05051)
"""
function window_convolution(W::Array{T,4}, v::Matrix) where {T}
    return @tullio C[i, k] := W[i, j, k, l] * v[j, l]
end

"""
    window_convolution(W::AbstractMatrix, v::AbstractVector)

Performs matrix-vector multiplication, where the matrix `W` acts as a linear
transformation or window applied to the vector input `v`.

# Arguments
- `W`: An abstract matrix representing the linear transformation or window.
- `v`: An abstract vector representing the input data.

# Returns
An abstract vector representing the result of the matrix-vector multiplication.

# Details
This method is a direct implementation of standard matrix-vector multiplication. It
applies the linear transformation defined by matrix `W` to the vector `v`.

# Formula
The operation is defined as:
```math
\\mathbf{c} = \\mathbf{W} \\mathbf{v}
```
or element-wise:
```math
c_i = \\sum_j W_{ij} v_j
```

# See Also
- [`window_convolution(W::Array{T, 4}, v::Matrix) where {T}`](@ref): Method for a 4D kernel and matrix input.

# References
- The methodology for this type of window measurement is discussed in: [arXiv:1810.05051](https://arxiv.org/abs/1810.05051)
"""
function window_convolution(W::AbstractMatrix, v::AbstractVector)
    return W * v
end
