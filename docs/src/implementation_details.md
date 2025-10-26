# Implementation Details

This page provides technical details about the algorithms and numerical methods implemented in `Effort.jl`.

## Overview

The package implements efficient computational methods for:

1. **Background cosmology** - ODE solvers for growth factors
2. **Neural network emulation** - Fast prediction of power spectrum multipoles
3. **Alcock-Paczynski corrections** - Geometric distortions from cosmological parameters
4. **Window convolution** - Survey geometry effects

---

## 1. Background Cosmology and Growth Factors

### Normalized Hubble Parameter

The normalized Hubble parameter ``E(z)`` quantifies the expansion rate of the universe as a function of redshift:

```math
E(z) = \frac{H(z)}{H_0} = \sqrt{\Omega_m(1+z)^3 + \Omega_k(1+z)^2 + \Omega_{DE}(z)}
```

For the w0waCDM model, the dark energy density parameter evolves as:

```math
\Omega_{DE}(z) = \Omega_{DE,0} (1+z)^{3(1 + w_0 + w_a)} \exp\left(-\frac{3w_a z}{1+z}\right)
```

### Growth Factor D(z)

The linear growth factor ``D(z)`` describes the growth of matter perturbations and is computed by solving the ODE:

```math
\frac{d^2 D}{da^2} + \left(\frac{3}{a} + \frac{d\ln H}{da}\right)\frac{dD}{da} = \frac{3\Omega_m(a)}{2a^2} D
```

where ``a = 1/(1+z)`` is the scale factor. This is solved using high-order Runge-Kutta methods (Tsit5) from `OrdinaryDiffEq.jl`.

**Initial conditions**: At high redshift (``z = 1000``), the growth factor is normalized such that ``D(z=0) = 1`` in the fiducial cosmology.

### Growth Rate f(z)

The logarithmic derivative of the growth factor is the growth rate:

```math
f(z) = \frac{d \ln D}{d \ln a} = -(1+z) \frac{dD/dz}{D}
```

This parameter appears in redshift-space distortions and affects the quadrupole and hexadecapole multipoles.

---

## 2. Power Spectrum Emulation

### Neural Network Architecture

The emulators use feedforward neural networks to predict power spectrum components:

- **Input**: 9 cosmological parameters ``\{z, \ln(10^{10}A_{\mathrm{s}}), n_{\mathrm{s}}, H_0, \omega_b, \omega_{\mathrm{cdm}}, \Sigma m_\nu, w_0, w_a\}``
- **Hidden layers**: Multiple hidden layers with activation functions
- **Output**: Three components of the power spectrum:
  - ``P_{11}``: Linear-linear contribution
  - ``P_{\text{loop}}``: One-loop contribution
  - ``P_{ct}``: Counter-term contribution

### Bias Expansion

The observed galaxy power spectrum is related to the matter power spectrum through bias parameters ``b_i``:

```math
P_g(k, \mu) = D^2 \sum_{i=1}^{N_{\text{bias}}} b_i \cdot \mathcal{B}_i(k, \mu; P_{11}, P_{\text{loop}}, P_{ct})
```

where ``\mathcal{B}_i`` are basis functions combining the emulated components with geometric factors.

**Standard setup**: 11 bias parameters, including:
- ``b_1``: Linear bias
- ``b_2, b_3, \ldots``: Higher-order bias terms
- ``b_8 = f(z)``: Growth rate (for RSD)
- ``b_9, b_{10}, b_{11}``: Stochastic and shot noise terms

---

## 3. Alcock-Paczynski Effect

### Physical Interpretation

The Alcock-Paczynski (AP) effect arises when the assumed reference cosmology differs from the true cosmology. Observations are made in redshift space ``(k_o, \mu_o)``, but the true clustering is in real space ``(k_t, \mu_t)``.

### AP Parameters

The distortion is quantified by two parameters:

```math
q_\parallel(z) = \frac{H_{\text{ref}}(z)}{H_{\text{true}}(z)} = \frac{E_{\text{ref}}(z)}{E_{\text{true}}(z)}
```

```math
q_\perp(z) = \frac{d_{A,\text{true}}(z)}{d_{A,\text{ref}}(z)} = \frac{\tilde{d}_{A,\text{true}}(z)}{\tilde{d}_{A,\text{ref}}(z)}
```

where ``d_A`` is the angular diameter distance and ``\tilde{d}_A`` is the conformal angular diameter distance.

**Physical meaning**:
- ``q_\parallel < 1``: Reference cosmology overestimates radial distances → clustering appears compressed along line-of-sight
- ``q_\perp < 1``: Reference cosmology overestimates transverse distances → clustering appears compressed perpendicular to line-of-sight

### Coordinate Transformation

The transformation from observed to true coordinates:

```math
k_t = \frac{k_o}{q_\perp} \sqrt{1 + \mu_o^2 \left(\frac{q_\perp^2}{q_\parallel^2} - 1\right)}
```

```math
\mu_t = \frac{\mu_o q_\perp}{q_\parallel \sqrt{1 + \mu_o^2 \left(\frac{q_\perp^2}{q_\parallel^2} - 1\right)}}
```

### Observed Power Spectrum

The observed power spectrum in redshift space is:

```math
P_{\text{obs}}(k_o, \mu_o) = \frac{1}{q_\parallel q_\perp^2} P_{\text{true}}(k_t, \mu_t)
```

The factor ``1/(q_\parallel q_\perp^2)`` accounts for the volume distortion.

### Multipole Projection

To obtain the observed multipoles, we integrate over the angular dependence:

```math
P_\ell(k_o) = (2\ell + 1) \int_0^1 P_{\text{obs}}(k_o, \mu_o) \mathcal{L}_\ell(\mu_o) \, d\mu_o
```

where ``\mathcal{L}_\ell(\mu)`` are Legendre polynomials:

```math
\begin{aligned}
\mathcal{L}_0(\mu) &= 1 \\
\mathcal{L}_2(\mu) &= \frac{1}{2}(3\mu^2 - 1) \\
\mathcal{L}_4(\mu) &= \frac{1}{8}(35\mu^4 - 30\mu^2 + 3)
\end{aligned}
```

### Numerical Implementation

`Effort.jl` provides two implementations:

#### 1. Standard Implementation (`apply_AP`)

Uses **Gauss-Lobatto quadrature** for fast integration:

1. Compute quadrature nodes ``\mu_1, \ldots, \mu_n`` and weights ``w_1, \ldots, w_n``
2. For each ``k_o`` and each ``\mu_i``:
   - Transform to true coordinates: ``(k_t, \mu_t) = f(k_o, \mu_i; q_\parallel, q_\perp)``
   - Interpolate true multipoles at ``k_t`` using Akima splines
   - Reconstruct ``P_{\text{true}}(k_t, \mu_t)`` from multipoles
   - Scale by volume factor: ``P_{\text{obs}} = P_{\text{true}} / (q_\parallel q_\perp^2)``
3. Compute weighted sum: ``P_\ell(k_o) = (2\ell+1) \sum_i w_i P_{\text{obs}}(k_o, \mu_i) \mathcal{L}_\ell(\mu_i)``

**Typical performance**: ~30 μs for 3 multipoles at 50 k-points (8 Gauss-Lobatto nodes)

**Exploiting symmetry**: Since the integrand is even in ``\mu`` for even multipoles (``\ell = 0, 2, 4``), we only integrate over ``[0, 1]`` instead of ``[-1, 1]``, halving the number of evaluations.

#### 2. Check Implementation (`apply_AP_check`)

Uses adaptive quadrature (QuadGK) for validation:

- Higher accuracy (``\text{reltol} = 10^{-12}``)
- ~10-100× slower than Gauss-Lobatto
- Useful for testing and verification

---

## 4. Interpolation: Akima Splines

Multipole moments are defined on a discrete k-grid, but AP corrections require evaluation at arbitrary ``k_t`` values. `Effort.jl` uses **Akima interpolation** for this purpose.

### Why Akima?

Compared to cubic splines:
- **Local**: Each interval depends only on 5 nearby points (vs. global for cubic splines)
- **No oscillations**: Avoids Runge's phenomenon near steep gradients
- **Fast**: Efficient for repeated evaluations

### Mathematical Form

For data points ``(k_1, P_1), \ldots, (k_n, P_n)``, Akima constructs a piecewise cubic polynomial:

```math
P(k) = a_i + b_i(k - k_i) + c_i(k - k_i)^2 + d_i(k - k_i)^3, \quad k \in [k_i, k_{i+1}]
```

Coefficients are determined by local slopes that minimize oscillations.

**Extrapolation**: Uses nearest-neighbor extension (constant extrapolation beyond data range).

---

## 5. Window Convolution

Survey geometry introduces correlations between different ``k`` modes through the window function ``W(k, k')``:

```math
P_{\ell}^{\text{obs}}(k) = \sum_{k'} W_{\ell\ell'}(k, k') P_{\ell'}^{\text{true}}(k')
```

For 2D analyses (``k`` and multipole ``\ell``), this generalizes to a 4D kernel:

```math
C_{ik} = \sum_{jl} W_{ijkl} v_{jl}
```

This is efficiently computed using the `@tullio` macro for tensor contractions.

**Applications**:
- Fiber collisions
- Angular selection functions
- Survey boundaries

**Reference**: [Beutler et al. 2019 (arXiv:1810.05051)](https://arxiv.org/abs/1810.05051)

---

## 6. Differentiation: Automatic vs Analytical

`Effort.jl` provides two complementary differentiation strategies, each optimized for specific use cases in cosmological inference.

### 6.1 Automatic Differentiation (AD)

The package is fully compatible with Julia's AD ecosystem:

- **ForwardDiff.jl**: Forward-mode AD for efficient gradients (explicitly tested)
- **Zygote.jl**: Reverse-mode AD for large parameter spaces (explicitly tested)

**Use case**: Gradient-based MCMC (HMC, NUTS) and maximum likelihood optimization.

When performing MCMC or variational inference, you can differentiate the likelihood function directly through the entire pipeline (ODE solvers → emulators → AP corrections). This works seamlessly because:

- **Custom ChainRules**: Hand-written adjoints for Akima interpolation, window convolution, and other critical operations
- **SciMLSensitivity**: Efficient gradients through ODE solvers via sensitivity analysis
- **Non-mutating code**: All functions avoid in-place mutations (Zygote-compatible)

**Example derivatives**:
```math
\frac{\partial \mathcal{L}}{\partial \theta_i} \quad \text{where } \theta \in \{h, \omega_c, w_0, \ldots\}
```

### 6.2 Analytical Jacobians

The package also provides **analytical Jacobian implementations** for derivatives with respect to bias parameters:

```math
J_{ki} = \frac{\partial P_\ell(k)}{\partial b_i}
```

**Use case**: Fisher Information Matrix computation for Jeffreys priors and survey forecasts.

**Why not use AD?** When computing Fisher matrices during MCMC analysis, using AD for Jacobians would require:

1. Computing Jacobian via AD: ``J = \nabla_{b} P_\ell(k; b)``
2. Differentiating likelihood (which uses ``J``) via AD: ``\nabla_{\theta} \mathcal{L}(\theta; J)``

This is **AD over AD** (nested differentiation), which is:
- Computationally expensive
- Numerically unstable
- Difficult to debug

**Solution**: Analytical Jacobians avoid this by providing closed-form derivatives. These are:

- **Fast**: Direct computation without AD overhead
- **Memory-efficient**: Optimized matrix operations
- **Validated**: Tested against both Computer Algebra Systems (symbolic differentiation) and AD (numerical validation)

The validation ensures:
```math
J_{\text{analytical}} \approx J_{\text{AD}} \approx J_{\text{CAS}} \quad (\text{within numerical precision})
```

### 6.3 Implementation Details

**AD-compatible operations**:
- Neural network evaluation (matrix operations)
- ODE solvers with SciMLSensitivity
- Akima interpolation with custom ChainRules
- Window convolution via `Tullio.jl` (AD-aware)

**Analytical Jacobian features**:
- Batched computation for all 11 bias parameters simultaneously
- Matrix interface compatible with `apply_AP` for efficient AP corrections
- Explicit tests in `test/test_pipeline.jl` validate against ForwardDiff, Zygote, and FiniteDifferences

---

## 7. Performance Optimizations

### Batch Processing

The `apply_AP` function supports matrix inputs, allowing multiple columns (e.g., Jacobian columns) to be processed simultaneously:

```julia
apply_AP(k_in, k_out, mono_matrix, quad_matrix, hexa_matrix, q_par, q_perp)
```

**Speedup**: ~2.5× faster than column-by-column processing by using optimized matrix Akima interpolation.

### Memory Efficiency

- Preallocated arrays for intermediate computations
- In-place operations where possible (avoiding allocations)
- Broadcasting for vectorized operations

### Numerical Stability

- Logarithmic extrapolation for power spectra (prevents negative values)
- Relative tolerances adapted to physical scales
- Robust handling of edge cases (``q \approx 1``, ``\mu \approx 0``)

---

## Summary

| Feature | Method | Performance |
|---------|--------|-------------|
| Growth factors D(z), f(z) | ODE solve (Tsit5) | ~160-180 μs |
| Multipole emulation | Neural network | ~26 μs per multipole |
| AP corrections | Gauss-Lobatto quadrature | ~33 μs (3 multipoles) |
| Interpolation | Akima splines | <1 μs per evaluation |
| Window convolution | Tensor contraction | Varies with kernel size |

All timings are approximate and depend on hardware (see Example page for system specifications).

---

## Further Reading

- **EFTofLSS theory**: [Baumann et al. 2012 (arXiv:1004.2488)](https://arxiv.org/abs/1004.2488)
- **PyBird code**: [D'Amico et al. 2020 (arXiv:1909.05271)](https://arxiv.org/abs/1909.05271)
- **Alcock-Paczynski effect**: [Alcock & Paczynski 1979](https://ui.adsabs.harvard.edu/abs/1979Natur.281..358A)
- **Neural network emulation**: [Nishimichi et al. 2019 (arXiv:1811.09504)](https://arxiv.org/abs/1811.09504)
- **Window convolution**: [Beutler et al. 2019 (arXiv:1810.05051)](https://arxiv.org/abs/1810.05051)
