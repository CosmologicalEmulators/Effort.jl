"""
    _transformed_weights(quadrature_rule, order, a, b)

Transforms the points and weights of a standard quadrature rule from the interval `[-1, 1]`
to a specified interval `[a, b]`.

This is a utility function used to adapt standard quadrature rules (like Gauss-Legendre)
for numerical integration over arbitrary intervals `[a, b]`.

# Arguments
- `quadrature_rule`: A function that takes an `order` and returns a tuple `(points, weights)`
                     for the standard interval `[-1, 1]`.
- `order`: The order of the quadrature rule (number of points).
- `a`: The lower bound of the target interval.
- `b`: The upper bound of the target interval.

# Returns
A tuple `(transformed_points, transformed_weights)` for the interval `[a, b]`.

# Details
The transformation is applied to the standard points `` x_i^{\\text{std}} `` and weights `` w_i^{\\text{std}} ``
obtained from the `quadrature_rule`:
- Transformed points: `` x_i = \\frac{b - a}{2} x_i^{\\text{std}} + \\frac{b + a}{2} ``
- Transformed weights: `` w_i = \\frac{b - a}{2} w_i^{\\text{std}} ``

# Formula
The transformation formulas are:
Points: `` x_i = \\frac{b - a}{2} x_i^{\\text{std}} + \\frac{b + a}{2} ``
Weights: `` w_i = \\frac{b - a}{2} w_i^{\\text{std}} ``

# See Also
- [`_r̃_z`](@ref): An example function that uses this utility for numerical integration.
"""
function _transformed_weights(quadrature_rule, order, a, b)
    x, w = quadrature_rule(order)
    x = (b - a) / 2.0 .* x .+ (b + a) / 2.0
    w = (b - a) / 2.0 .* w
    return x, w
end

function _akima_slopes(u, t)
    n = length(u)
    dt = diff(t)
    m = zeros(eltype(u[1] + t[1]), n + 3)
    m[3:(end-2)] = diff(u) ./ dt
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[end-1] = 2m[end-2] - m[end-3]
    m[end] = 2m[end-1] - m[end-2]
    return m
end

function _akima_coefficients(t, m)
    n = length(t)
    dt = diff(t)
    b = (m[4:end] .+ m[1:(end-3)]) ./ 2
    dm = abs.(diff(m))
    f1 = dm[3:(n+2)]
    f2 = dm[1:n]
    f12 = f1 + f2
    b = (f1 .* m[2:(end-2)] .+ f2 .* m[3:(end-1)]) ./ f12
    c = (3 .* m[3:(end-2)] .- 2 .* b[1:(end-1)] .- b[2:end]) ./ dt
    d = (b[1:(end-1)] .+ b[2:end] .- 2 .* m[3:(end-2)]) ./ dt .^ 2
    return b, c, d
end

function _akima_find_interval(t, tq)
    n = length(t)
    if tq ≤ t[1]
        return 1
    elseif tq ≥ t[end]
        return n - 1
    else
        idx = searchsortedlast(t, tq)
        return idx == n ? n - 1 : idx
    end
end

function _akima_eval(u, t, b, c, d, tq)
    idx = _akima_find_interval(t, tq)
    wj = tq - t[idx]
    return ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]
end

function _akima_eval(u, t, b, c, d, tq::AbstractArray)
    map(tqi -> _akima_eval(u, t, b, c, d, tqi), tq)
end

"""
    _akima_spline_legacy(u, t, t_new)

Evaluates the one-dimensional Akima spline that interpolates the data points ``(t_i, u_i)``
at new abscissae `t_new`.

# Arguments
- `u`: Ordinates (function values) ``u_i`` at the data nodes.
- `t`: Strictly increasing abscissae (knots) ``t_i`` associated with `u`. `length(t)` must equal `length(u)`.
- `t_new`: The query point(s) where the spline is to be evaluated.

# Returns
The interpolated value(s) at `t_new`. A scalar input returns a scalar; a vector input returns a vector of the same length.

# Details
This routine implements the original Akima piecewise-cubic method (T. Akima, 1970). On each interval ``[t_j, t_{j+1}]``, a cubic polynomial is constructed. The method uses a weighted average of slopes to determine the derivative at each node, which effectively dampens oscillations without explicit shape constraints. The resulting spline is ``C^1`` continuous (its first derivative is continuous) but generally not ``C^2``.

# Formulae
The spline on the interval ``[t_j, t_{j+1}]`` is a cubic polynomial:
\\[
S_j(w) = u_j + b_j w + c_j w^{2} + d_j w^{3}, \\qquad w = t - t_j
\\]
The derivative ``b_j`` at each node is determined by Akima's weighting of local slopes ``m_j=(u_{j}-u_{j-1})/(t_j-t_{j-1})``:
\\[
b_j = \\frac{|m_{j+1}-m_{j}|\\,m_{j-1} + |m_{j-1}-m_{j-2}|\\,m_{j}}
            {|m_{j+1}-m_{j}| + |m_{j-1}-m_{j-2}|}
\\]
The remaining coefficients, ``c_j`` and ``d_j``, are found by enforcing continuity of the first derivative:
\\[
c_j = \\frac{3m_j - 2b_j - b_{j+1}}{t_{j+1}-t_j}
\\]
\\[
d_j = \\frac{b_j + b_{j+1} - 2m_j}{(t_{j+1}-t_j)^2}
\\]

# Automatic Differentiation
The implementation is free of mutation on the inputs and uses only element-wise arithmetic, making the returned value differentiable with both `ForwardDiff.jl` (dual numbers) and `Zygote.jl` (reverse-mode AD). You can therefore embed `_akima_spline_legacy` in optimization or machine-learning pipelines and back-propagate through the interpolation seamlessly.

# Notes
The algorithm and numerical results are equivalent to the Akima spline in `DataInterpolations.jl`, but this routine is self-contained and avoids any package dependency.
"""
function _akima_spline_legacy(u, t, t_new)
    n = length(t)
    dt = diff(t)

    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)

    return _akima_eval(u, t, b, c, d, t_new)
end

"""
    _cubic_spline(u, t, new_t::AbstractArray)

A convenience wrapper to create and apply a cubic spline interpolation using `DataInterpolations.jl`.

This function simplifies the process of creating a `CubicSpline` interpolant for the data
`(u, t)` and evaluating it at the points `new_t`.

# Arguments
- `u`: An array of data values.
- `t`: An array of data points corresponding to `u`.
- `new_t`: An array of points at which to interpolate.

# Returns
An array of interpolated values corresponding to `new_t`.

# Details
This function is a convenience wrapper around `DataInterpolations.CubicSpline(u, t; extrapolation=ExtrapolationType.Extension).(new_t)`.
It creates a cubic spline interpolant with extrapolation enabled using `ExtrapolationType.Extension`
and immediately evaluates it at all points in `new_t`.

# See Also
- `DataInterpolations.CubicSpline`: The underlying interpolation function.
- [`_quadratic_spline`](@ref): Wrapper for quadratic spline interpolation.
- [`_akima_spline`](@ref): Wrapper for Akima interpolation.
"""
function _cubic_spline(u, t, new_t::AbstractArray)
    return DataInterpolations.CubicSpline(u, t; extrapolation=ExtrapolationType.Extension).(new_t)
end

"""
    _quadratic_spline(u, t, new_t::AbstractArray)

A convenience wrapper to create and apply a quadratic spline interpolation using `DataInterpolations.jl`.

This function simplifies the process of creating a `QuadraticSpline` interpolant for the data
`(u, t)` and evaluating it at the points `new_t`.

# Arguments
- `u`: An array of data values.
- `t`: An array of data points corresponding to `u`.
- `new_t`: An array of points at which to interpolate.

# Returns
An array of interpolated values corresponding to `new_t`.

# Details
This function is a convenience wrapper around `DataInterpolations.QuadraticSpline(u, t; extrapolation=ExtrapolationType.Extension).(new_t)`.
It creates a quadratic spline interpolant with extrapolation enabled using `ExtrapolationType.Extension`
and immediately evaluates it at all points in `new_t`.

# See Also
- `DataInterpolations.QuadraticSpline`: The underlying interpolation function.
- [`_cubic_spline`](@ref): Wrapper for cubic spline interpolation.
- [`_akima_spline`](@ref): Wrapper for Akima interpolation.
"""
function _quadratic_spline(u, t, new_t::AbstractArray)
    return DataInterpolations.QuadraticSpline(u, t; extrapolation=ExtrapolationType.Extension).(new_t)
end

"""
    _akima_spline(u, t, new_t::AbstractArray)

A convenience wrapper to create and apply an Akima interpolation using `DataInterpolations.jl`.

This function simplifies the process of creating an `AkimaInterpolation` interpolant for the data
`(u, t)` and evaluating it at the points `new_t`.

# Arguments
- `u`: An array of data values.
- `t`: An array of data points corresponding to `u`.
- `new_t`: An array of points at which to interpolate.

# Returns
An array of interpolated values corresponding to `new_t`.

# Details
This function is a convenience wrapper around `DataInterpolations.AkimaInterpolation(u, t; extrapolation=ExtrapolationType.Extension).(new_t)`.
It creates an Akima interpolant with extrapolation enabled using `ExtrapolationType.Extension`
and immediately evaluates it at all points in `new_t`.

# See Also
- `DataInterpolations.AkimaInterpolation`: The underlying interpolation function.
- [`_cubic_spline`](@ref): Wrapper for cubic spline interpolation.
- [`_quadratic_spline`](@ref): Wrapper for quadratic spline interpolation.
"""
function _akima_spline(u, t, new_t::AbstractArray)
    return DataInterpolations.AkimaInterpolation(u, t; extrapolation=ExtrapolationType.Extension).(new_t)
end

function _compose(z, t, new_t, Cᵢ_list, s_new, i_list, σ)
    return map(i -> z[i_list[i]-1] * (new_t[i] - t[i_list[i]-1]) +
                    σ[i] * (new_t[i] - t[i_list[i]-1])^2 + Cᵢ_list[i], 1:s_new)
end

function _create_σ(z, t, i_list)
    return map(i -> 1 / 2 * (z[i] - z[i-1]) / (t[i] - t[i-1]), i_list)
end

function _create_Cᵢ_list(u, i_list)
    return map(i -> u[i-1], i_list)
end

function _create_i_list(t, new_t, s_new)
    return map(i -> min(max(2, FindFirstFunctions.searchsortedfirstcorrelated(t, new_t[i],
                firstindex(t) - 1)), length(t)), 1:s_new)
end

function _create_d(u, t, s, typed_zero)
    return map(i -> i == 1 ? typed_zero : 2 * (u[i] - u[i-1]) / (t[i] - t[i-1]), 1:s)
end

"""
    _Legendre_0(x)

Calculates the 0th order Legendre polynomial, `` \\mathcal{L}_0(x) ``.

# Arguments
- `x`: The input value (typically the cosine of an angle, -1 ≤ x ≤ 1).

# Returns
The value of the 0th order Legendre polynomial evaluated at `x`.

# Formula
The formula for the 0th order Legendre polynomial is:
```math
\\mathcal{L}_0(x) = 1
```

# See Also
- [`_Legendre_2`](@ref): Calculates the 2nd order Legendre polynomial.
- [`_Legendre_4`](@ref): Calculates the 4th order Legendre polynomial.
- [`_Pkμ`](@ref): A function that uses Legendre polynomials.
"""
function _Legendre_0(x)
    return 1.0
end

"""
    _Legendre_2(x)

Calculates the 2nd order Legendre polynomial, `` \\mathcal{L}_2(x) ``.

# Arguments
- `x`: The input value (typically the cosine of an angle, -1 ≤ x ≤ 1).

# Returns
The value of the 2nd order Legendre polynomial evaluated at `x`.

# Formula
The formula for the 2nd order Legendre polynomial is:
```math
\\mathcal{L}_2(x) = \\frac{1}{2} (3x^2 - 1)
```

# See Also
- [`_Legendre_0`](@ref): Calculates the 0th order Legendre polynomial.
- [`_Legendre_4`](@ref): Calculates the 4th order Legendre polynomial.
- [`_Pkμ`](@ref): A function that uses Legendre polynomials.
"""
function _Legendre_2(x)
    return 0.5 * (3 * x^2 - 1)
end

"""
    _Legendre_4(x)

Calculates the 4th order Legendre polynomial, `` \\mathcal{L}_4(x) ``.

# Arguments
- `x`: The input value (typically the cosine of an angle, -1 ≤ x ≤ 1).

# Returns
The value of the 4th order Legendre polynomial evaluated at `x`.

# Formula
The formula for the 4th order Legendre polynomial is:
```math
\\mathcal{L}_4(x) = \\frac{1}{8} (35x^4 - 30x^2 + 3)
```

# See Also
- [`_Legendre_0`](@ref): Calculates the 0th order Legendre polynomial.
- [`_Legendre_2`](@ref): Calculates the 2nd order Legendre polynomial.
- [`_Pkμ`](@ref): A function that uses Legendre polynomials.
"""
function _Legendre_4(x)
    return 0.125 * (35 * x^4 - 30x^2 + 3)
end

function load_component_emulator(path::String; emu=LuxEmulator,
    k_file="k.npy", weights_file="weights.npy", inminmax_file="inminmax.npy",
    outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json",
    postprocessing_file="postprocessing_file.jl")

    # Load configuration for the neural network emulator
    NN_dict = parsefile(path * nn_setup_file)

    # Load the grid, emulator weights, and min-max scaling data
    kgrid = npzread(path * k_file)
    weights = npzread(path * weights_file)
    in_min_max = npzread(path * inminmax_file)
    out_min_max = npzread(path * outminmax_file)

    # Initialize the emulator using Effort.jl's init_emulator function
    trained_emu = Effort.init_emulator(NN_dict, weights, emu)

    # Instantiate and return the AbstractComponentEmulators struct
    return ComponentEmulator(
        TrainedEmulator=trained_emu,
        kgrid=kgrid,
        InMinMax=in_min_max,
        OutMinMax=out_min_max,
        Postprocessing=include(path * postprocessing_file)
    )
end

function load_multipole_emulator(path; emu=LuxEmulator,
    k_file="k.npy", weights_file="weights.npy", inminmax_file="inminmax.npy",
    outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json",
    postprocessing_file="postprocessing.jl", stochmodel_file="stochmodel.jl",
    biascombination_file="biascombination.jl", jacbiascombination_file="jacbiascombination.jl")

    P11 = load_component_emulator(path * "11/"; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file,
        postprocessing_file=postprocessing_file)

    Ploop = load_component_emulator(path * "loop/"; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file,
        postprocessing_file=postprocessing_file)

    Pct = load_component_emulator(path * "ct/"; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file,
        postprocessing_file=postprocessing_file)

    stochmodel = include(path * stochmodel_file)
    biascombination = include(path * biascombination_file)
    jacbiascombination = include(path * jacbiascombination_file)

    return PℓEmulator(P11=P11, Ploop=Ploop, Pct=Pct, StochModel=stochmodel,
        BiasCombination=biascombination, JacobianBiasCombination=jacbiascombination)
end
