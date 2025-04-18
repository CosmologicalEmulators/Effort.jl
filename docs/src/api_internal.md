# API reference

This section documents the functions intended for internal usage by the package.

## Index

```@index
Pages = ["api_internal.md"]
```

## Background

```@docs
Effort._F
Effort._dFdy
Effort._get_y
Effort._ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
Effort._ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
Effort._dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
Effort._dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
Effort._a_z
Effort._ρDE_a
Effort._ρDE_z
Effort._dρDEda
Effort._E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._E_a(a, w0wacosmo::Effort.w0waCDMCosmology)
Effort._E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._E_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._dlogEdloga
Effort._Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._Ωma(a, w0wacosmo::Effort.w0waCDMCosmology)
Effort._r̃_z_check
Effort._r̃_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._r̃_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._r_z_check(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._r_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._d̃A_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._dA_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._growth!
Effort._growth_solver(Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._growth_solver(w0wacosmo::Effort.w0waCDMCosmology)
Effort._growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._growth_solver(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._D_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._D_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
Effort._D_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._f_z(z::AbstractVector, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
Effort._f_z(z, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
Effort._f_z(z, w0wacosmo::Effort.w0waCDMCosmology)
Effort._D_f_z(z, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
Effort._D_f_z(z, w0wacosmo::Effort.w0waCDMCosmology)
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
Effort.apply_AP_check(k_input::Array, k_output::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par, q_perp)
Effort._Pk_recon
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
```
