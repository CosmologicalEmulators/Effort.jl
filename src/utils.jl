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

function _akima_spline_legacy(u::AbstractVector,
    t::AbstractVector,
    t_new)

    n = length(u)

    # 1. interval slopes  Δy/Δx
    s = _interval_slopes(u, t)          # length n-1

    # 2. ghost slopes for uniform treatment of endpoints
    s_ext = _extend_slopes(s)           # length n+3

    # 3. node derivatives (modified-Akima / makima)
    d = _node_derivatives_makima(s_ext) # length n

    # 4. evaluate spline
    return _akima_eval(u, t, d, t_new)
end

function _interval_slopes(u::AbstractVector, t::AbstractVector)
    n = length(u)
    s = zeros(eltype(u[1] - t[1]), n - 1)
    @inbounds for i in 1:n-1
        s[i] = (u[i+1] - u[i]) / (t[i+1] - t[i])
    end
    return s
end

function _extend_slopes(s::AbstractVector)
    # mirror the first and last two slopes to create 2 ghost values each side
    return vcat(2s[1] - s[2],
        2s[1] - s[2],
        s,
        2s[end] - s[end-1],
        2s[end] - s[end-1])
end

function _node_derivatives_makima(s_ext::AbstractVector)
    n = length(s_ext) - 3 # Number of derivatives to calculate
    d = zeros(eltype(s_ext), n)

    # The loop calculates the derivative d[i] for each node i.
    for i in 1:n
        # These are the weights used to average the slopes.
        w1 = (abs(s_ext[i+3] - s_ext[i+2]) + abs(s_ext[i+2] - s_ext[i+1]))

        w2 = if i == 1
            # For the first point, the original code accesses s_ext[0].
            # Based on the padding in `extend_slopes`, s_ext[1] == s_ext[2].
            # This implies a constant slope extrapolation, so we can assume
            # s_ext[0] would also equal s_ext[1].
            # The formula for w2 is: abs(s_ext[i+1]-s_ext[i]) + abs(s_ext[i]-s_ext[i-1])
            # For i=1, this becomes: abs(s_ext[2]-s_ext[1]) + abs(s_ext[1]-s_ext[0])
            # The first term is 0 because s_ext[1] == s_ext[2].
            # The second term becomes 0 under the constant slope assumption.
            0.0
        else
            # For all other points, the access is valid.
            (abs(s_ext[i+1] - s_ext[i]) + abs(s_ext[i] - s_ext[i-1]))
        end

        # The same issue exists at the end of the array for w1.
        # Let's add a similar check for the last point.
        if i == n
            # The formula for w1 is: abs(s_ext[i+3]-s_ext[i+2]) + abs(s_ext[i+2]-s_ext[i+1])
            # For i=n, this is: abs(s_ext[n+3]-s_ext[n+2]) + abs(s_ext[n+2]-s_ext[n+1])
            # The padding sets s_ext[n+2] == s_ext[n+3], making the first term 0.
            w1 = abs(s_ext[n+2] - s_ext[n+1])
        end

        if w1 + w2 == 0
            # If weights are zero, use a simple average of the adjacent slopes.
            # s_i is at s_ext[i+2], s_{i-1} is at s_ext[i+1].
            d[i] = (s_ext[i+2] + s_ext[i+1]) / 2
        else
            # The derivative is a weighted average of s_{i-1} and s_i.
            # In the original code, the numerator is `w1*s_ext[i+1] + w2*s_ext[i+2]`.
            d[i] = (w1 * s_ext[i+1] + w2 * s_ext[i+2]) / (w1 + w2)
        end
    end
    return d
end

function _akima_eval(u::AbstractVector, t::AbstractVector,
    d::AbstractVector, x::Real)
    n = length(u)
    i = clamp(searchsortedlast(t, x), 1, n - 1)   # locate interval
    h = t[i+1] - t[i]
    ξ = (x - t[i]) / h                          # normalised position

    # cubic Hermite basis
    h00 = 2ξ^3 - 3ξ^2 + 1
    h10 = ξ^3 - 2ξ^2 + ξ
    h01 = -2ξ^3 + 3ξ^2
    h11 = ξ^3 - ξ^2

    return h00 * u[i] + h10 * h * d[i] +
           h01 * u[i+1] + h11 * h * d[i+1]
end

function _get_i_list(t::AbstractVector, x::AbstractVector)
    n = length(t)
    return map(myx -> clamp(searchsortedlast(t, myx), 1, n - 1), x)
end

function _akima_eval(u::AbstractVector, t::AbstractVector,
    d::AbstractVector, x::AbstractVector)
    m = length(x)
    ilist = _get_i_list(t, x)   # locate interval
    h = map(i -> t[i+1] - t[i], ilist)
    ξ = map(i -> (x[i] - t[ilist[i]]) / h[i], 1:m)# normalised position

    # cubic Hermite basis
    ξ2 = ξ .^ 2
    ξ3 = ξ .^ 3

    h00 = 2 .* ξ3 .- 3 .* ξ2 .+ 1
    h10 = ξ3 .- 2 .* ξ2 .+ ξ
    h01 = -2 .* ξ3 .+ 3 .* ξ2
    h11 = ξ3 .- ξ2

    return map(i -> h00[i] * u[ilist[i]] + h10[i] * h[i] * d[ilist[i]] + h01[i] * u[ilist[i]+1] + h11[i] * h[i] * d[ilist[i]+1], 1:m)
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
