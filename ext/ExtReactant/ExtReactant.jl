__precompile__(false)

module ExtReactant

using Effort
using Reactant
using AbstractCosmologicalEmulators

import Effort: InterpolationMethod, ComponentEmulator, PℓEmulator
import AbstractCosmologicalEmulators: to_reactant

# Keep interpolation method tags as plain host values while tracing.
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:InterpolationMethod}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    return T
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::InterpolationMethod),
    @nospecialize(path),
    mode;
    kwargs...,
)
    return prev
end

"""
    to_reactant(comp::ComponentEmulator)

Move numeric arrays inside an Effort component emulator onto the active
Reactant device so they enter compiled graphs as traced inputs.
"""
function to_reactant(comp::ComponentEmulator)
    return ComponentEmulator(
        TrainedEmulator = to_reactant(comp.TrainedEmulator),
        # Keep wrapper arrays on host: ComponentEmulator type parameters require
        # AbstractMatrix{Float64}/AbstractArray{Float64}, while traced arrays do
        # not satisfy those constraints under Reactant tracing.
        kgrid = comp.kgrid,
        InMinMax = comp.InMinMax,
        OutMinMax = comp.OutMinMax,
        Postprocessing = comp.Postprocessing,
    )
end

"""
    to_reactant(emu::PℓEmulator)

Recursively convert an Effort multipole emulator for Reactant compilation.
Function fields are preserved; inner trained emulator payloads are moved to device.
"""
function to_reactant(emu::PℓEmulator)
    return PℓEmulator(
        P11 = to_reactant(emu.P11),
        Ploop = to_reactant(emu.Ploop),
        Pct = to_reactant(emu.Pct),
        StochModel = emu.StochModel,
        BiasCombination = emu.BiasCombination,
        JacobianBiasCombination = emu.JacobianBiasCombination,
    )
end

# Reactant-safe component evaluation: avoid host materialization
# (`Array(...)`) of traced outputs inside Effort.get_component.
function Effort.get_component(
    input_params::Reactant.TracedRArray{T,1},
    D,
    comp_emu::Effort.AbstractComponentEmulators,
) where {T}
    norm_input = Effort.maximin(input_params, comp_emu.InMinMax)
    norm_output = Effort.run_emulator(norm_input, comp_emu.TrainedEmulator)
    output = Effort.inv_maximin(norm_output, comp_emu.OutMinMax)
    postprocessed_output = comp_emu.Postprocessing(input_params, output, D, comp_emu)
    return reshape(postprocessed_output, length(comp_emu.kgrid), :)
end

# Reactant-safe multipole evaluation using emulator-provided closure functions.
function Effort.get_Pℓ(
    cosmology::Reactant.TracedRArray{T,1},
    D,
    bs::Reactant.TracedRArray{T,1},
    cosmoemu::Effort.AbstractPℓEmulators;
    stoch_kwargs...,
) where {T}
    P11_comp_array = Effort.get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = Effort.get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = Effort.get_component(cosmology, D, cosmoemu.Pct)
    stoch_comp_array = cosmoemu.StochModel(cosmoemu.P11.kgrid; stoch_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, stoch_comp_array)
    biases = cosmoemu.BiasCombination(bs)
    return stacked_array * biases
end


@inline _to_tracedish_vector(x::Reactant.TracedRArray) = x
@inline _to_tracedish_vector(x::Reactant.ConcretePJRTArray) = x
@inline _to_tracedish_vector(x::AbstractVector{<:Reactant.TracedRNumber}) = Reactant.stack(x)

@inline _to_tracedish_matrix(x::Reactant.TracedRArray) = x
@inline _to_tracedish_matrix(x::Reactant.ConcretePJRTArray) = x
@inline _to_tracedish_matrix(x::AbstractMatrix{<:Reactant.TracedRNumber}) = Reactant.stack(eachcol(x))

@inline function _safe_matvec_ext(A::AbstractMatrix, v::AbstractVector)
    # Reactant-safe matrix-vector product.
    # For large traced matrices, generic `A * v` may lower through typed_vcat
    # and blow up inference; this reduction form remains stable.
    return dropdims(sum(A .* reshape(v, 1, :), dims=2), dims=2)
end

function Effort.apply_AP(
    k_input::AbstractVector,
    k_output::AbstractVector,
    mono::AbstractVector{T},
    quad::AbstractVector{T},
    hexa::AbstractVector{T},
    q_par,
    q_perp;
    n_GL_points=8,
    method::InterpolationMethod=Effort.Cubic(),
) where {T}
    mono = _to_tracedish_vector(mono)
    quad = _to_tracedish_vector(quad)
    hexa = _to_tracedish_vector(hexa)

    mono = _to_tracedish_matrix(mono)
    quad = _to_tracedish_matrix(quad)
    hexa = _to_tracedish_matrix(hexa)

    nk = length(k_output)
    nodes, weights = Effort.gausslobatto(n_GL_points * 2)
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    F = q_par / q_perp

    k_t = Effort._k_true(k_output, μ_nodes, q_perp, F)
    μ_t = Effort._μ_true(μ_nodes, F)

    Pl0_t = Effort._Legendre_0.(μ_t)
    Pl2_t = Effort._Legendre_2.(μ_t)
    Pl4_t = Effort._Legendre_4.(μ_t)

    Pl0 = Effort._Legendre_0.(μ_nodes) .* μ_weights .* (2 * 0 + 1)
    Pl2 = Effort._Legendre_2.(μ_nodes) .* μ_weights .* (2 * 2 + 1)
    Pl4 = Effort._Legendre_4.(μ_nodes) .* μ_weights .* (2 * 4 + 1)

    new_mono_flat, new_quad_flat, new_hexa_flat = if method isa Effort.Cubic
        (
            AbstractCosmologicalEmulators.cubic_spline_interpolation(mono, k_input, k_t),
            AbstractCosmologicalEmulators.cubic_spline_interpolation(quad, k_input, k_t),
            AbstractCosmologicalEmulators.cubic_spline_interpolation(hexa, k_input, k_t),
        )
    elseif method isa Effort.Akima
        (
            AbstractCosmologicalEmulators.akima_interpolation(mono, k_input, k_t),
            AbstractCosmologicalEmulators.akima_interpolation(quad, k_input, k_t),
            AbstractCosmologicalEmulators.akima_interpolation(hexa, k_input, k_t),
        )
    else
        Effort._interpolate_multipoles(method, k_input, k_t, mono, quad, hexa)
    end

    new_mono = reshape(new_mono_flat, nk, n_GL_points)
    new_quad = reshape(new_quad_flat, nk, n_GL_points)
    new_hexa = reshape(new_hexa_flat, nk, n_GL_points)

    Pkμ = Effort._Pk_recon(new_mono, new_quad, new_hexa, Pl0_t, Pl2_t, Pl4_t) ./ (q_par * q_perp^2)

    return _safe_matvec_ext(Pkμ, Pl0), _safe_matvec_ext(Pkμ, Pl2), _safe_matvec_ext(Pkμ, Pl4)
end

function Effort.apply_AP(
    k_input::AbstractVector,
    k_output::AbstractVector,
    mono::AbstractMatrix{T},
    quad::AbstractMatrix{T},
    hexa::AbstractMatrix{T},
    q_par,
    q_perp;
    n_GL_points=8,
    method::InterpolationMethod=Effort.Cubic(),
) where {T}
    n_cols = size(mono, 2)
    nk = length(k_output)
    nodes, weights = Effort.gausslobatto(n_GL_points * 2)
    μ_nodes = nodes[1:n_GL_points]
    μ_weights = weights[1:n_GL_points]
    F = q_par / q_perp

    k_t = Effort._k_true(k_output, μ_nodes, q_perp, F)
    μ_t = Effort._μ_true(μ_nodes, F)

    Pl0_t = Effort._Legendre_0.(μ_t)
    Pl2_t = Effort._Legendre_2.(μ_t)
    Pl4_t = Effort._Legendre_4.(μ_t)

    Pl0 = Effort._Legendre_0.(μ_nodes) .* μ_weights .* (2 * 0 + 1)
    Pl2 = Effort._Legendre_2.(μ_nodes) .* μ_weights .* (2 * 2 + 1)
    Pl4 = Effort._Legendre_4.(μ_nodes) .* μ_weights .* (2 * 4 + 1)

    new_mono_flat, new_quad_flat, new_hexa_flat = if method isa Effort.Cubic
        (
            AbstractCosmologicalEmulators.cubic_spline_interpolation(mono, k_input, k_t),
            AbstractCosmologicalEmulators.cubic_spline_interpolation(quad, k_input, k_t),
            AbstractCosmologicalEmulators.cubic_spline_interpolation(hexa, k_input, k_t),
        )
    elseif method isa Effort.Akima
        (
            AbstractCosmologicalEmulators.akima_interpolation(mono, k_input, k_t),
            AbstractCosmologicalEmulators.akima_interpolation(quad, k_input, k_t),
            AbstractCosmologicalEmulators.akima_interpolation(hexa, k_input, k_t),
        )
    else
        Effort._interpolate_multipoles(method, k_input, k_t, mono, quad, hexa)
    end

    new_mono = reshape(new_mono_flat, nk, n_GL_points, n_cols)
    new_quad = reshape(new_quad_flat, nk, n_GL_points, n_cols)
    new_hexa = reshape(new_hexa_flat, nk, n_GL_points, n_cols)

    Pkμ = (new_mono .* reshape(Pl0_t, 1, :, 1) .+
           new_quad .* reshape(Pl2_t, 1, :, 1) .+
           new_hexa .* reshape(Pl4_t, 1, :, 1)) ./ (q_par * q_perp^2)

    Pkμ_reshaped = reshape(permutedims(Pkμ, (1, 3, 2)), nk * n_cols, n_GL_points)

    mono_out = reshape(_safe_matvec_ext(Pkμ_reshaped, Pl0), nk, n_cols)
    quad_out = reshape(_safe_matvec_ext(Pkμ_reshaped, Pl2), nk, n_cols)
    hexa_out = reshape(_safe_matvec_ext(Pkμ_reshaped, Pl4), nk, n_cols)

    return mono_out, quad_out, hexa_out
end

end # module ExtReactant
