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

"""
    akima_slopes(u, t)

Compute the "m" slopes for Akima interpolation, including extrapolated endpoints.
- `u`: values
- `t`: abscissae

Returns: `m::Vector` of size `length(u)+3`
"""
function akima_slopes(u, t)
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

"""
    akima_coefficients(u, t, m)

Given values `u`, abscissae `t`, and extended slopes `m`, compute (b, c, d) Akima coefficients.
Returns: `b, c, d` (all vectors)
"""
function akima_coefficients(t, m)
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

"""
    akima_find_interval(t, tq)

Find index of interval for `tq` in t.
Returns: the interval index (for left endpoint).
"""
function akima_find_interval(t, tq)
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

"""
    akima_eval(u, t, b, c, d, tq)

Evaluate Akima spline at a single point `tq`.
"""
function akima_eval(u, t, b, c, d, tq)
    idx = akima_find_interval(t, tq)
    wj = tq - t[idx]
    return ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]
end

function akima_eval(u, t, b, c, d, tq::AbstractArray)
    map(tqi -> akima_eval(u, t, b, c, d, tqi), tq)
end

"""
    akima_spline(u, t, t_new)

Main function: Akima spline as before, uses above kernels.
"""
function _akima_spline_legacy(u, t, t_new)
    n = length(t)
    dt = diff(t)

    m = akima_slopes(u, t)
    b, c, d = akima_coefficients(t, m)

    return akima_eval(u, t, b, c, d, t_new)
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

function load_component_emulator(path::String, comp_emu; emu=SimpleChainsEmulator,
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
    return comp_emu(
        TrainedEmulator=trained_emu,
        kgrid=kgrid,
        InMinMax=in_min_max,
        OutMinMax=out_min_max,
        Postprocessing=include(path * postprocessing_file)
    )
end

function load_multipole_emulator(path; emu=SimpleChainsEmulator,
    k_file="k.npy", weights_file="weights.npy", inminmax_file="inminmax.npy",
    outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json",
    postprocessing_file="postprocessing.jl", biascontraction_file="biascontraction.jl")

    P11 = load_component_emulator(path * "11/", Effort.P11Emulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file,
        postprocessing_file=postprocessing_file)

    Ploop = load_component_emulator(path * "loop/", Effort.PloopEmulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file,
        postprocessing_file=postprocessing_file)

    Pct = load_component_emulator(path * "ct/", Effort.PctEmulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file,
        postprocessing_file=postprocessing_file)

    biascontraction = include(path * biascontraction_file)

    return PℓEmulator(P11=P11, Ploop=Ploop, Pct=Pct, BiasContraction=biascontraction)
end

function load_multipole_noise_emulator(path; emu=SimpleChainsEmulator,
    k_file="k.npy", weights_file="weights.npy", inminmax_file="inminmax.npy",
    outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json")

    P11 = load_component_emulator(path * "11/", Effort.P11Emulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)

    Ploop = load_component_emulator(path * "loop/", Effort.PloopEmulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)

    Pct = load_component_emulator(path * "ct/", Effort.PctEmulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)

    Plemulator = PℓEmulator(P11=P11, Ploop=Ploop, Pct=Pct)

    NoiseEmulator = load_component_emulator(path * "st/", Effort.NoiseEmulator; emu=emu,
        k_file=k_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)

    return PℓNoiseEmulator(Pℓ=Plemulator, Noise=NoiseEmulator)
end
