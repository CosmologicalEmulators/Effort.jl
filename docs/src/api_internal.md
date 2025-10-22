# API reference

This section documents the functions intended for internal usage by the package.

## Index

```@index
Pages = ["api_internal.md"]
```

## Neural Networks

```@docs
Effort.AbstractComponentEmulators
Effort.ComponentEmulator
Effort.get_component
Effort.AbstractPℓEmulators
Effort.PℓEmulator
```

## EFT Commands

```@docs
Effort.get_Pℓ_jacobian
```

## Projection

```@docs
Effort._Pkμ
Effort._k_true(k_o, μ_o, q_perp, F)
Effort._k_true(k_o::Array, μ_o::Array, q_perp, F)
Effort._μ_true(μ_o, F)
Effort._μ_true(μ_o::Array, F)
Effort._P_obs
Effort.interp_Pℓs
Effort.q_par_perp
Effort.apply_AP_check(k_input::AbstractVector, k_output::AbstractVector, Mono_array::AbstractVector, Quad_array::AbstractVector, Hexa_array::AbstractVector, q_par, q_perp)
Effort.apply_AP(k_input::AbstractVector, k_output::AbstractVector, mono::AbstractVector, quad::AbstractVector, hexa::AbstractVector, q_par, q_perp)
Effort.apply_AP(k_input::AbstractVector, k_output::AbstractVector, mono::AbstractMatrix, quad::AbstractMatrix, hexa::AbstractMatrix, q_par, q_perp)
Effort._Pk_recon
Effort.window_convolution
```

## Utils

```@docs
Effort._transformed_weights
Effort._Legendre_0
Effort._Legendre_2
Effort._Legendre_4
Effort._cubic_spline
Effort._quadratic_spline
Effort._akima_spline
Effort._akima_spline_legacy(u, t, t_new)
Effort._akima_spline_legacy(u::AbstractMatrix, t, t_new)
Effort._akima_slopes(u::AbstractMatrix, t)
Effort._akima_coefficients(t, m::AbstractMatrix)
Effort._akima_eval(u::AbstractMatrix, t, b::AbstractMatrix, c::AbstractMatrix, d::AbstractMatrix, tq::AbstractArray)
Effort.load_component_emulator
Effort.load_multipole_emulator
```
