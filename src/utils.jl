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
"""
function _transformed_weights(quadrature_rule, order, a, b)
    x, w = quadrature_rule(order)
    x = (b - a) / 2.0 .* x .+ (b + a) / 2.0
    w = (b - a) / 2.0 .* w
    return x, w
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
    return one(x)
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
    T = typeof(x)
    return T(0.5) * (T(3) * x^2 - one(x))
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
    T = typeof(x)
    return T(0.125) * (T(35) * x^4 - T(30) * x^2 + T(3))
end

"""
    load_component_emulator(path::String; emu=LuxEmulator, k_file="k.npy", weights_file="weights.npy", inminmax_file="inminmax.npy", outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json", postprocessing_file="postprocessing_file.jl")

Load a trained component emulator from disk.

# Arguments
- `path::String`: Directory path containing the emulator files.

# Keyword Arguments
- `emu`: Emulator type to initialize (`LuxEmulator` or `SimpleChainsEmulator`). Default: `LuxEmulator`.
- `k_file::String`: Filename for the wavenumber grid. Default: `"k.npy"`.
- `weights_file::String`: Filename for neural network weights. Default: `"weights.npy"`.
- `inminmax_file::String`: Filename for input normalization parameters. Default: `"inminmax.npy"`.
- `outminmax_file::String`: Filename for output normalization parameters. Default: `"outminmax.npy"`.
- `nn_setup_file::String`: Filename for network architecture configuration. Default: `"nn_setup.json"`.
- `postprocessing_file::String`: Filename for postprocessing function. Default: `"postprocessing_file.jl"`.

# Returns
A `ComponentEmulator` instance ready for evaluation.

# Details
This function loads all necessary files to reconstruct a trained component emulator:
1. Neural network architecture from JSON configuration.
2. Trained weights from NumPy binary format.
3. Normalization parameters for inputs and outputs.
4. Wavenumber grid.
5. Postprocessing function dynamically loaded from Julia file.

The postprocessing function is evaluated in an isolated scope to prevent namespace pollution.

# Example
```julia
P11_emu = load_component_emulator("/path/to/emulator/11/")
```

# File Structure
The expected directory structure is:
```
path/
├── k.npy                    # Wavenumber grid
├── weights.npy              # Neural network weights
├── inminmax.npy            # Input normalization (n_params × 2)
├── outminmax.npy           # Output normalization (n_k × 2)
├── nn_setup.json           # Network architecture
└── postprocessing_file.jl  # Postprocessing function
```
"""
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
        Postprocessing=eval(Meta.parse("let; " * read(path * postprocessing_file, String) * " end"))
    )
end

"""
    load_multipole_emulator(path; emu=LuxEmulator, k_file="k.npy", weights_file="weights.npy", inminmax_file="inminmax.npy", outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json", postprocessing_file="postprocessing.jl", stochmodel_file="stochmodel.jl", biascombination_file="biascombination.jl", jacbiascombination_file="jacbiascombination.jl")

Load a complete power spectrum multipole emulator from disk.

# Arguments
- `path`: Directory path containing the multipole emulator structure.

# Keyword Arguments
- `emu`: Emulator type to initialize (`LuxEmulator` or `SimpleChainsEmulator`). Default: `LuxEmulator`.
- `k_file::String`: Filename for the wavenumber grid. Default: `"k.npy"`.
- `weights_file::String`: Filename for neural network weights. Default: `"weights.npy"`.
- `inminmax_file::String`: Filename for input normalization parameters. Default: `"inminmax.npy"`.
- `outminmax_file::String`: Filename for output normalization parameters. Default: `"outminmax.npy"`.
- `nn_setup_file::String`: Filename for network architecture configuration. Default: `"nn_setup.json"`.
- `postprocessing_file::String`: Filename for postprocessing function. Default: `"postprocessing.jl"`.
- `stochmodel_file::String`: Filename for stochastic model function. Default: `"stochmodel.jl"`.
- `biascombination_file::String`: Filename for bias combination function. Default: `"biascombination.jl"`.
- `jacbiascombination_file::String`: Filename for bias Jacobian function. Default: `"jacbiascombination.jl"`.

# Returns
A `PℓEmulator` instance containing all three components (P11, Ploop, Pct) and bias models.

# Details
This function loads a complete multipole emulator by:
1. Loading three component emulators (P11, Ploop, Pct) from subdirectories.
2. Loading the stochastic model function (shot noise terms).
3. Loading the bias combination function (maps bias parameters to weights).
4. Loading the analytical Jacobian of the bias combination.

All functions are evaluated in isolated scopes to prevent namespace conflicts between
different emulator components.

# Example
```julia
# Load monopole emulator
monopole_emu = load_multipole_emulator("/path/to/artifact/0/")

# Evaluate
cosmology = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
bias = [b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2]
D = 0.8
P0 = get_Pℓ(cosmology, D, bias, monopole_emu)
```

# File Structure
The expected directory structure is:
```
path/
├── 11/                       # P11 component
│   ├── k.npy
│   ├── weights.npy
│   ├── inminmax.npy
│   ├── outminmax.npy
│   ├── nn_setup.json
│   └── postprocessing.jl
├── loop/                     # Ploop component
│   └── ... (same structure)
├── ct/                       # Pct component
│   └── ... (same structure)
├── stochmodel.jl            # Stochastic model function
├── biascombination.jl       # Bias combination function
└── jacbiascombination.jl    # Bias Jacobian function
```

# See Also
- [`load_component_emulator`](@ref): Load individual component emulators.
- [`get_Pℓ`](@ref): Evaluate the loaded emulator.
"""
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

    # Load functions in isolated scopes to prevent name conflicts
    stochmodel = eval(Meta.parse("let; " * read(path * stochmodel_file, String) * " end"))
    biascombination = eval(Meta.parse("let; " * read(path * biascombination_file, String) * " end"))
    jacbiascombination = eval(Meta.parse("let; " * read(path * jacbiascombination_file, String) * " end"))

    return PℓEmulator(P11=P11, Ploop=Ploop, Pct=Pct, StochModel=stochmodel,
        BiasCombination=biascombination, JacobianBiasCombination=jacbiascombination)
end
