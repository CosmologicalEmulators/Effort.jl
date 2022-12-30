# Example

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using BenchmarkTools
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/effort_benchmark.json")
```

In order to use `Effort` you need a trained emulator (after the first release, we will make them available on Zenodo). There are two different categories of trained emulators:

- single component emulators (e.g.  $P_{11}$, $P_\mathrm{loop}$, $P_\mathrm{ct}$)
- complete emulators, containing all the three different component emulators

In this section we are going to show how to:

- obtain a multipole power spectrum, using a trained emulator
- apply the Alcock-Paczyński effect
- compute stochastic term contibution

## Basic usage

Let us show how to use Effort.jl to compute Power Spectrum Multipoles.

First of all, we need some trained emulators, then we can use the `Effort.get_Pℓ` function

```@docs
Effort.get_Pℓ
```

!!! info "Trained emulators"
    Right now we do not provide any emulator, but with the paper publication we will release
    several trained emulators on Zenodo.

```julia
import Effort
Pct_comp_array = Effort.compute_component(input_test, Pct_Mono_emu) #compute the components of Pct without the bias
Pct_array_Effort = Array{Float64}(zeros(length(Pct_comp_array[1,:]))) #allocate final array
Effort.bias_multiplication!(Pct_array_Effort, bct, Pct_comp_array) #components multiplied by bias
Effort.get_Pℓ(input_test, bs, f, Pℓ_Mono_emu) # whole multipole computation
```

Here we are using a `ComponentEmulators`, which can compute one of the components as
predicted by PyBird, and `MultipoleEmualator`, which emulates an entire multipole.

This computation is quite fast: a benchmark performed locally, with a 12th Gen Intel® Core™
i7-1260P, gives the following result for a multipole computation

```@example tutorial
benchmark[1]["Effort"]["Monopole"] # hide
```

The result of these computations look like this
![effort](https://user-images.githubusercontent.com/58727599/209453056-a83dfd18-03c2-46be-a3a5-01b5f3bd459d.png)

## Alcock-Paczyński effect

Here we are going to write down the equations related to the AP effect, following the [Ivanov et al. (2019)](https://arxiv.org/abs/1909.05277) and [D'Amico et al.- (2020)](https://arxiv.org/abs/2003.07956) notation.

In particular, we are going to use:

- ``\rm{ref}``, for the quantities evaluated in the reference cosmology used to perform the measurements
- ``\rm{true}``, for the quantities evaluated in the true cosmology used to perform the theoretical predictions
- ``\mathrm{obs}``, for the observed quantities after applying the AP effect

The wavenumbers parallel and perpendicular to the line of sight ``(k^\mathrm{true}_\parallel,\, k^\mathrm{true}_\perp)`` are related to the ones of the reference cosmology as ``(k^\mathrm{ref}_\parallel,\, k^\mathrm{ref}_\perp)`` as:

```math
k_{\|}^{\text {ref }}=q_{\|} k^\mathrm{true}_{\|}, \quad k_{\perp}^{\mathrm{ref}}=q_{\perp} k^\mathrm{true}_{\perp}
```

where the distortion parameters are defined by

```math
q_{\|}=\frac{D^\mathrm{true}_A(z) H^\mathrm{true}(z=0)}{D_A^{\mathrm{ref}}(z) H^{\mathrm{ref}}(z=0)}, \quad q_{\perp}=\frac{H^{\mathrm{ref}}(z) / H^{\mathrm{ref}}(z=0)}{H^\mathrm{true}(z) / H^\mathrm{true}(z=0)}
```

where ``D^A``, ``H`` are the angular diameter distance and Hubble parameter, respectively. In terms of these parameters, the power spectrum multipoles in the reference cosmology is given by the multipole projection integral

```math
P_{\ell, \mathrm{AP}}(k)=\frac{2 \ell+1}{2} \int_{-1}^1 d \mu_{\mathrm{obs}} P_{\mathrm{obs}}\left(k_{\mathrm{obs}}, \mu_{\mathrm{obs}}\right) \cdot \mathcal{P}_{\ell}\left(\mu_{\mathrm{obs}}\right)
```

The observed ``P_{\mathrm{obs}}\left(k_{\mathrm{obs}}, \mu_{\mathrm{obs}}\right)``, when including the AP effect, is given by

```math
P_{\mathrm{obs}}\left(k_{\mathrm{obs}}, \mu_{\mathrm{obs}}\right)= \frac{1}{q_{\|} q_{\perp}^2} \cdot P_g\left(k_{\text {true }}\left[k_{\mathrm{obs}}, \mu_{\mathrm{obs}}\right], \mu_{\text {true }}\left[k_{\text {obs }}, \mu_{\mathrm{obs}}\right]\right)
```

The Alcock-Paczyński (AP) effect can be included in two different ways:

- by training the emulators using spectra where the AP effect has already been applied
- by using standard trained emulators and applying analitycally the AP effect

While the former approach is computationally faster (there is no overhead from the NN
point-of-view), the latter is more flexible, since the reference cosmology for the AP effect
computation can be changed at runtime.

Regarding the second approach, the most important choice regards the algorithm employed to
compute the multipole projection integral.
Here we implement two different approaches, based on

- [QuadGK.jl](https://juliamath.github.io/QuadGK.jl/stable/). This approach is the most precise, since it uses an adaptive method to compute the integral.
- [FastGaussQuadrature.jl](https://juliaapproximation.github.io/FastGaussQuadrature.jl/stable/). This approach is the fastest, since we are going to employ only 4 points (!!!) to compute the integral, taking advantage of the Gauss-Lobatto quadrature rule.

Let us start with the Gauss-Kronrod quadrature!

```@docs
Effort.apply_AP_check
```

```julia
import Effort
Effort.apply_AP_check(k_test, Mono_Effort, Quad_Effort, Hexa_Effort,  q_par, q_perp)
```

```@example tutorial
benchmark[1]["Effort"]["AP_GK"] # hide
```

As said, this is precise but a bit expensive from a computational point of view. What about
Gauss-Lobatto?

```@docs
Effort.apply_AP
```

```julia
import Effort
Effort.apply_AP(k_test, Mono_Effort, Quad_Effort, Hexa_Effort,  q_par, q_perp)
```

```@example tutorial
benchmark[1]["Effort"]["AP_GL"] # hide
```

Blazingly fast! And also accurate! A comparison with the GK-based rule show a percentual
relative difference of about $0.00001\%$ for the Hexadecapole, with a higher precision for
the other two multipoles.

![gk_gl_residuals](https://user-images.githubusercontent.com/58727599/210023676-e040a484-1c04-483e-a88a-cd1ee925830f.png)
