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
Effort._ΩνE2(a::Number, Ωγ0::Number, mν::AbstractVector)
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
```

## Utils

```@docs
Effort._transformed_weights
```
