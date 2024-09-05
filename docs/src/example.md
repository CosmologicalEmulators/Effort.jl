# Example

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using BenchmarkTools
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/effort_benchmark.json")
new_benchmark = BenchmarkTools.load("./assets/new_effort.json")
```

In order to use `Effort.jl` you need a trained emulator. There are two different categories of trained emulators:

- single component emulators (e.g.  $P_{11}$, $P_\mathrm{loop}$, $P_\mathrm{ct}$)
- complete emulators, containing all the three different component emulators

In this section we are going to show how to:

- obtain a multipole power spectrum, using a trained emulator
- apply the Alcock-Paczyński effect
- compute stochastic term contribution

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

Here we are using a `ComponentEmulator`, which can compute one of the components as
predicted by PyBird, and a `MultipoleEmulator`, which emulates an entire multipole.

This computation is quite fast: a benchmark performed locally, with a 12th Gen Intel® Core™
i7-1260P, gives the following result for a multipole computation

```@example tutorial
benchmark[1]["Effort"]["Monopole"] # hide
```

The result of these computations look like this
![effort](https://user-images.githubusercontent.com/58727599/209453056-a83dfd18-03c2-46be-a3a5-01b5f3bd459d.png)

## Alcock-Paczyński effect

Here we are going to write down the equations related to the AP effect, following the [Ivanov et al. (2019)](https://arxiv.org/abs/1909.05277) and [D'Amico et al. (2020)](https://arxiv.org/abs/2003.07956) notation.

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

In the `Effort.jl` workflow, the Alcock-Paczyński (AP) effect can be included in two different ways:

- by training the emulators using spectra where the AP effect has already been applied
- by using standard trained emulators and applying analitycally the AP effect

While the former approach is computationally faster (there is no overhead from the NN
point-of-view), the latter is more flexible, since the reference cosmology for the AP effect
computation can be changed at runtime.

Regarding the second approach, the most important choice regards the algorithm employed to
compute the multipole projection integral.
Here we implement two different approaches, based on

- [QuadGK.jl](https://juliamath.github.io/QuadGK.jl/stable/). This approach is the most precise, since it uses an adaptive method to compute the integral.
- [FastGaussQuadrature.jl](https://juliaapproximation.github.io/FastGaussQuadrature.jl/stable/). This approach is the fastest, since we are going to employ only 5 points to compute the integral, taking advantage of the Gauss-Lobatto quadrature rule.

In order to understand _why_ it is possible to use few points to evaluate the AP projection integral, it is intructive to plot the ``\mu`` dependence of the integrand

![mu_dependence](https://user-images.githubusercontent.com/58727599/210108594-8c2c1c02-22e9-4d5d-a266-5fffa92bbcba.png)

The ``\ell=4`` integrand, the most complicated one, can be accurately fit with a ``n=8`` polynomial

![polyfit_residuals](https://user-images.githubusercontent.com/58727599/210109373-fbd9ab7e-1926-4761-a972-8045724b6704.png)

Since a ``n`` Gauss-Lobatto rule can integrate exactly ``2n – 3`` degree polynomials,  we expect that a GL rule with 10 points can perform the integral with high precision.

Now we can show how to use Effort.jl to compute the AP effect using the GK adaptive integration

```@docs
Effort.apply_AP_check
```

```julia
import Effort
Effort.apply_AP_check(k_test, Mono_Effort, Quad_Effort, Hexa_Effort,  q_par, q_perp)
```

```@example tutorial
new_benchmark[1]["Effort"]["AP_check"] # hide
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
new_benchmark[1]["Effort"]["AP"] # hide
```

This is 200 times faster than the adaptive integration, but is also very accurate! A
comparison with the GK-based rule shows a percentual relative difference of about
$10^{-11}\%$ for the Hexadecapole, with a higher precision for the other two multipoles.

![gk_gl_residuals](https://user-images.githubusercontent.com/58727599/210110289-ec61612c-5ef2-4691-87fb-386f186f5e5e.png)

## Growth factor

A quantity required to compute EFTofLSS observables is the growth rate, ``f``. While other emulator packages employ an emulator also for ``f`` (or equivalently emulate the growth factor ``D``), we choose a different approach, using the [DiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/) library to efficiently solve the equation for the growth factor, as written in [Bayer, Banerjee & Feng (2021)](https://arxiv.org/abs/2007.13394)

```math
D''(a)+\left(2+\frac{E'(a)}{E(a)}\right)D'(a)=\frac{3}{2}\Omega_{m}(a)D(a),
```

where ``E(a)`` is the adimensional Hubble factor, whose expression is given by

```math
E(a)=\left[\Omega_{\gamma, 0} a^{-4}+\Omega_{c, 0} a^{-3}+\Omega_\nu(a) E^2(a)+\Omega_{\mathrm{DE}}(a)\right]^{1 / 2}
```

Since we start solving the equation deep in the matter dominated era, when ``D(a)\sim a``, we can set as initial conditions

```math
D(z_i) = a_i
```

```math
D'(z_i)=a_i
```

In ``E(a)``, we precisely take into account radiation, non-relativistic matter, massive neutrinos, evolving Dark Energy.
Regarding massive neutrinos, their energy density is given by

```math
\Omega_\nu(a) E^2(a)=\frac{15}{\pi^4} \Gamma_\nu^4 \frac{\Omega_{\gamma, 0}}{a^4} \sum_{j=1}^{N_\nu} \mathcal{F}\left(\frac{m_j a}{k_B T_{\nu, 0}}\right)
```

with

```math
\mathcal{F}(y) \equiv \int_0^{\infty} d x \frac{x^2 \sqrt{x^2+y^2}}{1+e^x}
```

Regarding Dark Energy, its contribution to the Hubble is

```math
\Omega_{\mathrm{DE}}(a)=\Omega_\mathrm{DE,0}(1+z)^{3\left(1+w_0+w_a\right)} e^{-3 w_a z /(1+z)}
```

Solving the previous equation is quite fast, as the benchmark shows

```julia
@benchmark Effort._D_z($z, $ΩM, $h)
```

```@example tutorial
new_benchmark[1]["Effort"]["Growth"] # hide
```

The result is also quite accurate; here is a check against the CLASS computation both for
the growth factor and the growth rate

![growth_check_class](https://user-images.githubusercontent.com/58727599/210219849-09646729-365a-4ab9-9372-d72e0a808c78.png)

Since the final goal is to embedd `Effort` in bayesian analysis pipelines which need gradient computations, emphasis has been put on its compatibility with AD tools such as `ForwardDiff` and `Enzyme`. In particular, for the ODE solution, this is guaranteed by the `SciMLSensitivity` stack.

Comparing with Fig. 5 of [Donald-McCann et al. (2021)](https://arxiv.org/abs/2109.15236), we see that the error is similar to the one they obtained, with the advantage that we don't have the restriction of an emulation range. However, if required, we may as well include an emulator for ``D(z)`` and ``f(z)``.

## Automatic Differentiation

Great care has been devoted to ensure that `Effort` is compatible with AD systems. Here, in particular, we are going to show the performance of backward-AD as implemented in `Zygote`.

```julia
@benchmark Zygote.gradient(k_test->sum(Effort.apply_AP(k_test, Mono_Effort, Quad_Effort, Hexa_Effort,  q_par, q_perp)), k_test)
```

```@example tutorial
new_benchmark[1]["Effort"]["AP & Zygote"] # hide
```
