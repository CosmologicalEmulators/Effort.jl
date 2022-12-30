```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using BenchmarkTools
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/effort_benchmark.json")
```

# Example

In order to use `Effort` you need a trained emulator (after the first release, we will make them available on Zenodo). There are two different categories of trained emulators:

- single component emulators (e.g.  $P_{11}$, $P_\mathrm{loop}$, $P_\mathrm{ct}$)
- complete emulators, containing all the three different component emulators

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
The Alcock-Paczyński (AP) effect can be included in two different ways:

- by training the emulators using spectra where the AP effect has already been applied
- by using standard trained emulators and applying analitycally the AP effect

While the former approach is computationally faster (there is no overhead from the NN
point-of-view), the latter is more flexible, since the reference cosmology for the AP effect
computation can be changed at runtime.

Regarding the second approach, the most important choice regards the algorithm employed to
compute the following integral

```math
P_{\ell, \mathrm{AP}}(k)=\frac{2 \ell+1}{2} \int_{-1}^1 d \mu_{\mathrm{obs}} P_{\mathrm{obs}}\left(k_{\mathrm{obs}}, \mu_{\mathrm{obs}}\right) \cdot \mathcal{P}_{\ell}\left(\mu_{\mathrm{obs}}\right)
```

Here we implement two different approaches:

- an approach based on [QuadGK.jl](https://juliamath.github.io/QuadGK.jl/stable/). This approach is the most precise, since it uses an adaptive method to compute the AP projection integral.
- an approach based on [FastGaussQuadrature.jl](https://juliaapproximation.github.io/FastGaussQuadrature.jl/stable/). This approach is the fastest, since we are going to employ only 4 points (!!!) to compute the AP projection integral, taking advantage of the Gauss-Lobatto quadrature rule
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
