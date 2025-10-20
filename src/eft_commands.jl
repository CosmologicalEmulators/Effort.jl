"""
    get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; stoch_kwargs...)

Compute the power spectrum multipole `` P_\\ell(k) `` given cosmological parameters, bias parameters,
and growth factor.

# Arguments
- `cosmology::Array`: Array of cosmological parameters (format depends on the emulator training).
- `D`: Growth factor value at the redshift of interest.
- `bs::Array`: Array of bias parameters.
- `cosmoemu::AbstractPℓEmulators`: The multipole emulator containing P11, Ploop, Pct components.

# Keyword Arguments
- `stoch_kwargs...`: Additional keyword arguments passed to the stochastic model (e.g., shot noise parameters).

# Returns
- Power spectrum multipole values evaluated on the emulator's k-grid.

# Details
This function computes the power spectrum by:
1. Evaluating the P11, Ploop, and Pct components using the trained neural network emulators.
2. Computing the stochastic contribution.
3. Combining all components via the bias combination function.

"""
function get_Pℓ(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; stoch_kwargs...)

    P11_comp_array = get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pct)
    stoch_comp_array = cosmoemu.StochModel(cosmoemu.P11.kgrid; stoch_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, stoch_comp_array)
    biases = cosmoemu.BiasCombination(bs)

    return stacked_array * biases
end

"""
    get_Pℓ_jacobian(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; stoch_kwargs...)

Compute both the power spectrum multipole `` P_\\ell(k) `` and its Jacobian with respect to
bias parameters.

# Arguments
- `cosmology::Array`: Array of cosmological parameters (format depends on the emulator training).
- `D`: Growth factor value at the redshift of interest.
- `bs::Array`: Array of bias parameters.
- `cosmoemu::AbstractPℓEmulators`: The multipole emulator containing P11, Ploop, Pct components.

# Keyword Arguments
- `stoch_kwargs...`: Additional keyword arguments passed to the stochastic model (e.g., shot noise parameters).

# Returns
A tuple `(Pℓ, ∂Pℓ_∂b)` where:
- `Pℓ`: Power spectrum multipole values evaluated on the emulator's k-grid.
- `∂Pℓ_∂b`: Jacobian matrix of the power spectrum with respect to bias parameters.

# Details
This function is optimized for inference workflows where both the power spectrum and its
derivatives are needed (e.g., gradient-based MCMC, Fisher forecasts). It computes both
quantities in a single pass, avoiding redundant neural network evaluations.

The Jacobian is computed using the analytical derivative of the bias combination function,
which is significantly faster than automatic differentiation for this specific operation.

# See Also
- [`get_Pℓ`](@ref): Compute only the power spectrum without Jacobian.
"""
function get_Pℓ_jacobian(cosmology::Array, D, bs::Array, cosmoemu::AbstractPℓEmulators; stoch_kwargs...)

    P11_comp_array = get_component(cosmology, D, cosmoemu.P11)
    Ploop_comp_array = get_component(cosmology, D, cosmoemu.Ploop)
    Pct_comp_array = get_component(cosmology, D, cosmoemu.Pct)
    stoch_comp_array = cosmoemu.StochModel(cosmoemu.P11.kgrid; stoch_kwargs...)
    stacked_array = hcat(P11_comp_array, Ploop_comp_array, Pct_comp_array, stoch_comp_array)
    biases = cosmoemu.BiasCombination(bs)
    jacbiases = cosmoemu.JacobianBiasCombination(bs)

    return stacked_array * biases, stacked_array * jacbiases
end
