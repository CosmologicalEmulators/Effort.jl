# External API

This section documents the public functions intended for users.

## Power Spectrum Computation

```@docs
Effort.get_Pℓ
Effort.get_Pℓ_jacobian
```

## Alcock-Paczynski Corrections

```@docs
Effort.q_par_perp
Effort.apply_AP
```

## Window Convolution

```@docs
Effort.window_convolution
```

## Unified AP + Window Pipeline

```@docs
Effort.APWindowChebyshevPlan
Effort.prepare_ap_window_chebyshev
Effort.apply_AP_and_window
```

## Chebyshev Operators

```@docs
Effort.ChebyshevOperator
Effort.prepare_chebyshev_operator
Effort.apply_chebyshev_operator
```
