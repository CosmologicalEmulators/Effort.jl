"""
    AbstractComponentEmulators

Abstract type for component-level emulators that represent individual parts of the
power spectrum calculation (e.g., P11, Ploop, Pct).

All concrete subtypes must implement the necessary fields to enable neural network
evaluation, normalization, and postprocessing.
"""
abstract type AbstractComponentEmulators end

"""
    ComponentEmulator <: AbstractComponentEmulators

A complete emulator for a single power spectrum component, combining neural network
predictions with normalization and physics-based postprocessing.

# Fields
- `TrainedEmulator::AbstractTrainedEmulators`: The trained neural network (Lux or SimpleChains).
- `kgrid::Array`: Wavenumber grid on which the component is evaluated (in h/Mpc).
- `InMinMax::Matrix{Float64}`: Min-max normalization parameters for inputs (n_params × 2).
- `OutMinMax::Array{Float64}`: Min-max normalization parameters for outputs (n_k × 2).
- `Postprocessing::Function`: Function to apply physics transformations to raw NN output.

# Details
The typical evaluation flow is:
1. Normalize input parameters using `InMinMax`.
2. Evaluate neural network to get normalized output.
3. Denormalize output using `OutMinMax`.
4. Apply postprocessing (e.g., multiply by D² for P11).

# Example Postprocessing
```julia
# For linear power spectrum component
postprocess_P11 = (params, output, D, emu) -> output .* D^2
```
"""
@kwdef struct ComponentEmulator <: AbstractComponentEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64}
    OutMinMax::Array{Float64}
    Postprocessing::Function
end

"""
    get_component(input_params, D, comp_emu::AbstractComponentEmulators)

Evaluate a component emulator to obtain power spectrum component values.

# Arguments
- `input_params`: Array of input parameters (e.g., cosmological parameters).
- `D`: Growth factor at the redshift of interest.
- `comp_emu::AbstractComponentEmulators`: The component emulator to evaluate.

# Returns
A matrix of shape `(n_k, n_samples)` containing the evaluated power spectrum component
values on the emulator's k-grid.

# Details
This function performs the full evaluation pipeline:
1. Copy input parameters to avoid mutation.
2. Apply min-max normalization to inputs.
3. Run neural network inference.
4. Denormalize network output.
5. Apply component-specific postprocessing (using `D` and emulator metadata).
6. Reshape to match k-grid dimensions.

The postprocessing step typically includes physics-based transformations such as
scaling by powers of the growth factor.
"""
function get_component(input_params, D, comp_emu::AbstractComponentEmulators)
    input = deepcopy(input_params)
    norm_input = maximin(input, comp_emu.InMinMax)
    norm_output = Array(run_emulator(norm_input, comp_emu.TrainedEmulator))
    output = inv_maximin(norm_output, comp_emu.OutMinMax)
    postprocessed_output = comp_emu.Postprocessing(input_params, output, D, comp_emu)
    return reshape(postprocessed_output, length(comp_emu.kgrid), :)
end

"""
    AbstractPℓEmulators

Abstract type for complete power spectrum multipole emulators.

Concrete subtypes must combine multiple component emulators (P11, Ploop, Pct) with
bias models to compute full power spectrum multipoles `` P_\\ell(k) `` for `` \\ell \\in \\{0, 2, 4\\} ``.
"""
abstract type AbstractPℓEmulators end

"""
    PℓEmulator <: AbstractPℓEmulators

Complete emulator for power spectrum multipoles in the Effective Field Theory of
Large Scale Structure (EFTofLSS) framework.

# Fields
- `P11::ComponentEmulator`: Emulator for the linear theory power spectrum component.
- `Ploop::ComponentEmulator`: Emulator for the one-loop corrections.
- `Pct::ComponentEmulator`: Emulator for the counterterm contributions.
- `StochModel::Function`: Function to compute stochastic (shot noise) terms.
- `BiasCombination::Function`: Function mapping bias parameters to linear combination weights.
- `JacobianBiasCombination::Function`: Analytical Jacobian of `BiasCombination` w.r.t. bias parameters.

# Details
The power spectrum multipole is computed as:
```math
P_\\ell(k) = \\sum_i c_i(b_1, b_2, ...) P_i(k)
```
where:
- `` P_i(k) `` are the component power spectra (P11, Ploop, Pct, stochastic terms)
- `` c_i(b_1, b_2, ...) `` are coefficients from the bias expansion

The `BiasCombination` function encodes the EFT bias model,
while `JacobianBiasCombination` provides analytical derivatives for efficient gradient-based
inference.

# Example Usage
```julia
# Load pre-trained emulator
emu = trained_emulators["PyBirdmnuw0wacdm"]["0"]  # monopole

# Evaluate power spectrum
cosmology = [z, ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
bias = [b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2]
D = 0.8  # growth factor

P0 = get_Pℓ(cosmology, D, bias, emu)
```

# See Also
- [`get_Pℓ`](@ref): Evaluate the power spectrum.
- [`get_Pℓ_jacobian`](@ref): Evaluate power spectrum and its Jacobian.
"""
@kwdef struct PℓEmulator <: AbstractPℓEmulators
    P11::ComponentEmulator
    Ploop::ComponentEmulator
    Pct::ComponentEmulator
    StochModel::Function
    BiasCombination::Function
    JacobianBiasCombination::Function
end
