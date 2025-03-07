function _transformed_weights(quadrature_rule, order, a,b)
    x, w = quadrature_rule(order)
    x = (b-a)/2. .* x .+ (b+a)/2.
    w = (b-a)/2. .* w
    return x, w
end

function _quadratic_spline_legacy(u, t, new_t::Number)
    s = length(t)
    dl = ones(eltype(t), s - 1)
    d_tmp = ones(eltype(t), s)
    du = zeros(eltype(t), s - 1)
    tA = Tridiagonal(dl, d_tmp, du)

    # zero for element type of d, which we don't know yet
    typed_zero = zero(2 // 1 * (u[begin + 1] - u[begin]) / (t[begin + 1] - t[begin]))

    d = map(i -> i == 1 ? typed_zero : 2 // 1 * (u[i] - u[i - 1]) / (t[i] - t[i - 1]), 1:s)
    z = tA \ d
    i = min(max(2, FindFirstFunctions.searchsortedfirstcorrelated(t, new_t, firstindex(t) - 1)), length(t))
    Cᵢ = u[i - 1]
    σ = 1 // 2 * (z[i] - z[i - 1]) / (t[i] - t[i - 1])
    return z[i - 1] * (new_t - t[i - 1]) + σ * (new_t - t[i - 1])^2 + Cᵢ
end

function _quadratic_spline_legacy(u, t, new_t::AbstractArray)
    s = length(t)
    s_new = length(new_t)
    dl = ones(eltype(t), s - 1)
    d_tmp = ones(eltype(t), s)
    du = zeros(eltype(t), s - 1)
    tA = Tridiagonal(dl, d_tmp, du)

    # zero for element type of d, which we don't know yet
    typed_zero = zero(2 // 1 * (u[begin + 1] - u[begin]) / (t[begin + 1] - t[begin]))

    d = _create_d(u, t, s, typed_zero)
    z = tA \ d
    i_list = _create_i_list(t, new_t, s_new)
    Cᵢ_list = _create_Cᵢ_list(u, i_list)
    σ = _create_σ(z, t, i_list)
    return _compose(z, t, new_t, Cᵢ_list, s_new, i_list, σ)
end

function _cubic_spline(u, t, new_t::AbstractArray)
    return DataInterpolations.CubicSpline(u,t; extrapolation = ExtrapolationType.Extension).(new_t)
end

function _quadratic_spline(u, t, new_t::AbstractArray)
    return DataInterpolations.QuadraticSpline(u,t; extrapolation = ExtrapolationType.Extension).(new_t)
end

function _akima_spline(u, t, new_t::AbstractArray)
    return DataInterpolations.AkimaInterpolation(u,t; extrapolation = ExtrapolationType.Extension).(new_t)
end

function _compose(z, t, new_t, Cᵢ_list, s_new, i_list, σ)
    return   map(i -> z[i_list[i] - 1] * (new_t[i] - t[i_list[i] - 1]) +
             σ[i] * (new_t[i] - t[i_list[i] - 1])^2 + Cᵢ_list[i], 1:s_new)
end

function _create_σ(z, t, i_list)
    return map(i -> 1 / 2 * (z[i] - z[i - 1]) / (t[i] - t[i - 1]),  i_list)
end

function _create_Cᵢ_list(u, i_list)
    return map(i-> u[i - 1],  i_list)
end

function _create_i_list(t, new_t, s_new)
    return map(i-> min(max(2, FindFirstFunctions.searchsortedfirstcorrelated(t, new_t[i],
    firstindex(t) - 1)), length(t)),  1:s_new)
end

function _create_d(u, t, s, typed_zero)
    return map(i -> i == 1 ? typed_zero : 2 * (u[i] - u[i - 1]) / (t[i] - t[i - 1]), 1:s)
end

function _legendre_0(x)
    return 1.
end

function _legendre_2(x)
    return 0.5*(3*x^2-1)
end

function _legendre_4(x)
    return 0.125*(35*x^4-30x^2+3)
end

function load_component_emulator(path::String, comp_emu; emu = SimpleChainsEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing_file.jl")

    # Load configuration for the neural network emulator
    NN_dict = parsefile(path * nn_setup_file)

    # Load the grid, emulator weights, and min-max scaling data
    kgrid = npzread(path * k_file)
    weights = npzread(path * weights_file)
    in_min_max = npzread(path * inminmax_file)
    out_min_max = npzread(path * outminmax_file)

    # Initialize the emulator using Capse.jl's init_emulator function
    trained_emu = Effort.init_emulator(NN_dict, weights, emu)

    # Instantiate and return the AbstractComponentEmulators struct
    return comp_emu(
        TrainedEmulator = trained_emu,
        kgrid = kgrid,
        InMinMax = in_min_max,
        OutMinMax = out_min_max,
        Postprocessing = include(path*postprocessing_file)
    )
end

function load_multipole_emulator(path; emu = SimpleChainsEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl", biascontraction_file = "biascontraction.jl")

    P11 = load_component_emulator(path*"11/", Effort.P11Emulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file,
    postprocessing_file = postprocessing_file)

    Ploop = load_component_emulator(path*"loop/", Effort.PloopEmulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file,
    postprocessing_file = postprocessing_file)

    Pct = load_component_emulator(path*"ct/", Effort.PctEmulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file,
    postprocessing_file = postprocessing_file)

    biascontraction = include(path*biascontraction_file)

    return PℓEmulator(P11=P11, Ploop=Ploop, Pct=Pct, BiasContraction = biascontraction)
end

function load_multipole_noise_emulator(path; emu = SimpleChainsEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json")

    P11 = load_component_emulator(path*"11/", Effort.P11Emulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file)

    Ploop = load_component_emulator(path*"loop/", Effort.PloopEmulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file)

    Pct = load_component_emulator(path*"ct/", Effort.PctEmulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file)

    Plemulator = PℓEmulator(P11=P11, Ploop=Ploop, Pct=Pct)

    NoiseEmulator = load_component_emulator(path*"st/", Effort.NoiseEmulator; emu = emu,
    k_file = k_file, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file)

    return PℓNoiseEmulator(Pℓ=Plemulator, Noise=NoiseEmulator)
end
