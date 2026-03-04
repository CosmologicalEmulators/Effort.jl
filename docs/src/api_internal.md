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

## Projection

```@docs
Effort._Pkμ
Effort._k_true
Effort._μ_true
Effort._P_obs
Effort.interp_Pℓs
Effort.apply_AP_check
Effort._Pk_recon
```

## Utils

```@docs
Effort._transformed_weights
Effort._Legendre_0
Effort._Legendre_2
Effort._Legendre_4
Effort.load_component_emulator
Effort.load_multipole_emulator
```
