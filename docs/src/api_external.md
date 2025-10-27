# External API

This section documents the public functions intended for users.

## Power Spectrum Computation

```@docs
Effort.get_Pℓ
Effort.get_Pℓ_jacobian
```

## Alcock-Paczynski Corrections

```@docs
Effort.q_par_perp(z, cosmo_mcmc::Effort.AbstractCosmology, cosmo_ref::Effort.AbstractCosmology)
Effort.apply_AP
```

## Window Convolution

```@docs
Effort.window_convolution(W::Array{T, 4}, v::Matrix) where {T}
Effort.window_convolution(W::AbstractMatrix, v::AbstractVector)
```
