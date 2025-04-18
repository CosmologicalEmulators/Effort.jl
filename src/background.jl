#TODO: this part should be moved to a dedicate package. While necessary to a full Effort
#functionality, this could be factorized to a new module, specifically taylored to this goal
# and maybe used in other packages, maybe in AbstractCosmologicalEmulators?

abstract type AbstractCosmology end


"""
    w0waCDMCosmology(ln10Aₛ::Number, nₛ::Number, h::Number, ωb::Number, ωc::Number, mν::Number=0., w0::Number=-1., wa::Number=0.)

This struct contains the value of the cosmological parameters for ``w_0 w_a``CDM cosmologies.

## Keyword arguments

 - `ln10Aₛ` and `nₛ`, the amplitude and the tilt of the primordial power spectrum fluctuations
 - `h`, the value of the reduced Hubble paramater
 - `ωb` and `ωc`, the physical energy densities of baryons and cold dark matter
 - `mν`, the sum of the neutrino masses in eV
 - `w₀` and `wₐ`, the Dark Energy equation of state parameters in the [CPL parameterization](https://arxiv.org/abs/astro-ph/0208512)
"""
@kwdef mutable struct w0waCDMCosmology <: AbstractCosmology
    ln10Aₛ::Number
    nₛ::Number
    h::Number
    ωb::Number
    ωc::Number
    mν::Number = 0.0
    w0::Number = -1.0
    wa::Number = 0.0
end

"""
    _F(y)

# Arguments
- `y`: The value of the parameter `y` for which the integral is calculated.

# Returns
The value of the definite integral for the given `y`.

# Details
The integrand is defined as:
``f(x, y) = x^2 \\cdot \\sqrt{x^2 + y^2} / (1 + e^x)``

The integration is performed over the domain `(0, Inf)` for the variable `x`.
A relative tolerance of `1e-12` is used for the integration solver.
"""
function _F(y)
    f(x, y) = x^2 * √(x^2 + y^2) / (1 + exp(x))
    domain = (zero(eltype(Inf)), Inf) # (lb, ub)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol
end

"""
    _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)

Calculates the dimensionless parameter `y` used in the integral function [`_F(y)`](@ref).

The parameter `y` is calculated based on the neutrino mass, scale factor,
Boltzmann constant, and neutrino temperature according to the formula:

`y = mν * a / (kB * Tν)`

# Arguments
- `mν`: Neutrino mass (in units where `kB` and `Tν` are defined).
- `a`: Scale factor.

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K).
- `Tν`: Neutrino temperature (default: 0.71611 * 2.7255 K).

# Returns
The calculated dimensionless parameter `y`.
"""
function _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)
    return mν * a / (kB * Tν)
end

"""
    _dFdy(y)

Calculates the definite integral of the function ``f(x, y) = x^2 / ((1 + e^x) \\cdot \\sqrt{x^2 + y^2})``
with respect to `x` from `0` to `Inf`, and then multiplies the result by `y`.

This function is the derivative of the integral function [`_F(y)`](@ref)
with respect to `y`.

# Arguments
- `y`: The value of the parameter `y` used in the integrand and as a multiplicative factor.

# Returns
The value of the definite integral multiplied by `y` for the given `y`.
"""
function _dFdy(y)
    f(x, y) = x^2 / ((1 + exp(x)) * √(x^2 + y^2))
    domain = (zero(eltype(Inf)), Inf) # (lb, ub)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol * y
end

"""
    _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculates the energy density of relic neutrinos, scaled by the critical density,
at a given scale factor `a`, for a *single* neutrino mass.

This function accounts for the contribution of a single neutrino mass `mν` to the total
energy density. It uses [`_F(y)`](@ref) to incorporate the effect of neutrino mass and temperature.

# Arguments
- `a`: The scale factor.
- `Ωγ0`: The photon density parameter today.
- `mν`: The neutrino mass (a single value).

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K).
- `Tν`: Neutrino temperature (default: 0.71611 * 2.7255 K).
- `Neff`: Effective number of neutrino species (default: 3.044).

# Returns
The calculated neutrino energy density parameter `ΩνE2` at scale factor `a` for the given mass.

# Details
The calculation involves a factor `Γν` derived from `Neff` and the ratio of
neutrino to photon temperatures. The main term is proportional to `Ωγ0 / a^4`
multiplied by `F_interpolant(_get_y(mν, a))`.

The parameter `y` passed to `F_interpolant` is calculated using [`_get_y(mν, a)`](@ref).

# Formula
The formula used is:
`ΩνE2 = (15 / π^4) * Γν^4 * (Ωγ0 / a^4) * F(y)`
where `Γν = (4/11)^(1/3) * (Neff/3)^(1/4)` and `y = mν * a / (kB * Tν)`.

# See Also
- [`_get_y(mν, a)`](@ref): Calculates the `y` parameter.
- [`_F(y)`](@ref): The integral function used as `F_interpolant`.
- [`_ΩνE2(a, Ωγ0, mν::AbstractVector)`](@ref): Method for a vector of neutrino masses.
"""
function _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * F_interpolant(_get_y(mν, a))
end

"""
    _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculates the energy density of relic neutrinos, scaled by the critical density,
at a given scale factor `a`, for a *vector* of neutrino masses.

This function accounts for the combined contribution of multiple neutrino masses
to the total energy density by summing the individual contributions. It uses the
`F_interpolant` function (which is equivalent to [`_F(y)`](@ref)) for each mass.

# Arguments
- `a`: The scale factor.
- `Ωγ0`: The photon density parameter today.
- `mν`: A vector of neutrino masses (`AbstractVector`).

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K).
- `Tν`: Neutrino temperature (default: 0.71611 * 2.7255 K).
- `Neff`: Effective number of neutrino species (default: 3.044).

# Returns
The calculated total neutrino energy density parameter `ΩνE2` at scale factor `a`
for the sum of contributions from all masses in the vector.

# Details
The calculation involves a factor `Γν` derived from `Neff` and the ratio of
neutrino to photon temperatures. The main term is proportional to `Ωγ0 / a^4`
multiplied by the sum of `F_interpolant(_get_y(mν_i, a))` for each mass `mν_i`
in the input vector `mν`.

The parameter `y` passed to `F_interpolant` for each mass is calculated using
[`_get_y(mν_i, a)`](@ref).

# Formula
The formula used is:
`ΩνE2 = (15 / π^4) * Γν^4 * (Ωγ0 / a^4) * Σ F(y_i)`
where `Γν = (4/11)^(1/3) * (Neff/3)^(1/4)` and `y_i = mν_i * a / (kB * Tν)`.

# See Also
- [`_get_y(mν, a)`](@ref): Calculates the `y` parameter for each mass.
- [`_F(y)`](@ref): The integral function used as `F_interpolant`.
- [`_ΩνE2(a, Ωγ0, mν)`](@ref): Method for a single neutrino mass.
"""
function _ΩνE2(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += F_interpolant(_get_y(mymν, a))
    end
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * sum_interpolant
end

"""
    _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculates the derivative of the neutrino energy density parameter [`_ΩνE2`](@ref)
with respect to the scale factor `a`, for a *single* neutrino mass.

This function computes the derivative of the expression for `_ΩνE2` by applying
the chain rule, involving both [`_F(y)`](@ref)
and [`_dFdy(y)`](@ref) functions.

# Arguments
- `a`: The scale factor.
- `Ωγ0`: The photon density parameter today.
- `mν`: The neutrino mass (a single value).

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K).
- `Tν`: Neutrino temperature (default: 0.71611 * 2.7255 K).
- `Neff`: Effective number of neutrino species (default: 3.044).

# Returns
The calculated derivative `d(ΩνE2)/da` at scale factor `a` for the given mass.

# Details
The calculation is based on the derivative of the `_ΩνE2` formula with respect to `a`.

# See Also
- [`_ΩνE2(a, Ωγ0, mν)`](@ref): The function whose derivative is calculated.
- [`_get_y(mν, a)`](@ref): Calculates the `y` parameter.
- [`_F(y)`](@ref): The integral function used as `F_interpolant`.
- [`_dFdy(y)`](@ref): The function used as `dFdy_interpolant`.
- [`_dΩνE2da(a, Ωγ0, mν::AbstractVector)`](@ref): Method for a vector of neutrino masses.
"""
function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 * (-4 * F_interpolant(_get_y(mν, a)) / a^5 +
                                    dFdy_interpolant(_get_y(mν, a)) / a^4 * (mν / kB / Tν))
end


"""
    _dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)

Calculates the derivative of the neutrino energy density parameter [`_ΩνE2`](@ref)
with respect to the scale factor `a`, for a *vector* of neutrino masses.

This function computes the derivative of the expression for `_ΩνE2` by summing
the derivatives of the contributions from each individual neutrino mass. It uses
the [`_F(y)`](@ref) and [`_dFdy(y)`](@ref) functions for each mass.

# Arguments
- `a`: The scale factor.
- `Ωγ0`: The photon density parameter today.
- `mν`: A vector of neutrino masses (`AbstractVector`).

# Keyword Arguments
- `kB`: Boltzmann constant (default: 8.617342e-5 eV/K).
- `Tν`: Neutrino temperature (default: 0.71611 * 2.7255 K).
- `Neff`: Effective number of neutrino species (default: 3.044).

# Returns
The calculated total derivative `d(ΩνE2)/da` at scale factor `a` for the sum
of contributions from all masses in the vector.

# Details
The calculation sums the derivatives of the individual neutrino mass contributions
to `_ΩνE2` with respect to `a`.

# See Also
- [`_ΩνE2(a, Ωγ0, mν::AbstractVector)`](@ref): The function whose derivative is calculated.
- [`_get_y(mν, a)`](@ref): Calculates the `y` parameter for each mass.
- [`_F(y)`](@ref): The integral function used as `F_interpolant`.
- [`_dFdy(y)`](@ref): The function used as `dFdy_interpolant`.
- [`_dΩνE2da(a, Ωγ0, mν)`](@ref): Method for a single neutrino mass.
"""
function _dΩνE2da(a, Ωγ0, mν::AbstractVector; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += -4 * F_interpolant(_get_y(mymν, a)) / a^5 +
                           dFdy_interpolant(_get_y(mymν, a)) / a^4 * (mymν / kB / Tν)
    end
    return 15 / π^4 * Γν^4 * Ωγ0 * sum_interpolant
end

function _a_z(z)
    return @. 1 / (1 + z)
end

function _E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return @. sqrt(Ωγ0 * a^-4 + Ωcb0 * a^-3 + ΩΛ0 * _ρDE_a(a, w0, wa) + _ΩνE2(a, Ωγ0, mν))
end

function _E_a(a, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _E_a(a, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    a = _a_z(z)
    return _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
end

function _E_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _E_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return a * 0.5 / (_E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)^2) *
           (-3(Ωcb0)a^-4 - 4Ωγ0 * a^-5 + ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν))
end

function _Ωma(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return Ωcb0 * a^-3 / (_E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa))^2
end

function _Ωma(a, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _Ωma(a, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _r̃_z_check(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    p = [Ωcb0, h, mν, w0, wa]
    f(x, p) = 1 / _E_a(_a_z(x), p[1], p[2]; mν=p[3], w0=p[4], wa=p[5])
    domain = (zero(eltype(z)), z) # (lb, ub)
    prob = IntegralProblem(f, domain, p; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol
end

function _r̃_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    z_array, weigths_array = _transformed_weights(FastGaussQuadrature.gausslegendre, 9, 0, z)
    integrand_array = 1.0 ./ _E_a(_a_z(z_array), Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return dot(weigths_array, integrand_array)
end

function _r̃_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _r̃_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _r_z_check(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _r̃_z_check(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

function _r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

function _r_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _r_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (1 + z)
end

function _d̃A_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _d̃A_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return _r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (1 + z)
end

function _dA_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _dA_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _ρDE_z(z, w0, wa)
    return (1 + z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z / (1 + z))
end

function _ρDE_a(a, w0, wa)
    return a^(-3.0 * (1.0 + w0 + wa)) * exp(3.0 * wa * (a - 1))
end

function _dρDEda(a, w0, wa)
    return 3 * (-(1 + w0 + wa) / a + wa) * _ρDE_a(a, w0, wa)
end

function _growth!(du, u, p, loga)
    Ωcb0 = p[1]
    mν = p[2]
    h = p[3]
    w0 = p[4]
    wa = p[5]
    a = exp(loga)
    D = u[1]
    dD = u[2]
    du[1] = dD
    du[2] = -(2 + _dlogEdloga(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)) * dD +
            1.5 * _Ωma(a, Ωcb0, h; mν=mν, w0=w0, wa=wa) * D
end

function _growth_solver(Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    u₀ = [amin, amin]

    logaspan = (log(amin), log(1.01))#to ensure we cover the relevant range

    p = [Ωcb0, mν, h, w0, wa]

    prob = ODEProblem(_growth!, u₀, logaspan, p)

    sol = solve(prob, Tsit5(), reltol=1e-5; verbose=false)
    return sol
end

function _growth_solver(w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _growth_solver(Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    loga = vcat(log.(_a_z.(z)))#, 0.0)# this is to ensure the *normalized version* is
    #properly normalized, if uncommented
    u₀ = [amin, amin]

    logaspan = (log(amin), log(1.01))#to ensure we cover the relevant range

    p = [Ωcb0, mν, h, w0, wa]

    prob = ODEProblem(_growth!, u₀, logaspan, p)

    sol = solve(prob, Tsit5(), reltol=1e-5; saveat=loga)[1:2, :]
    return sol
end

function _growth_solver(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _growth_solver(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _D_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return (sol(log(_a_z(z)))[1, :])[1, 1][1]
end

function _D_z(z::AbstractVector, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return reverse(sol[1, 1:end])
end

function _D_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _D_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _f_z(z::AbstractVector, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    D = sol[1, 1:end]
    D_prime = sol[2, 1:end]#if wanna have normalized_version, 1:end
    result = @. 1 / D * D_prime
    return reverse(result)
end

function _f_z(z, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    D = sol[1, 1:end][1]
    D_prime = sol[2, 1:end][1]
    return (1/D*D_prime)[1]
end

function _f_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _f_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end

function _D_f_z(z, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    D = sol[1, 1:end]
    D_prime = sol[2, 1:end]
    f = @. 1 / D * D_prime
    return reverse(D), reverse(f)
end

function _D_f_z(z, w0wacosmo::w0waCDMCosmology)
    Ωcb0 = (w0wacosmo.ωb + w0wacosmo.ωc) / w0wacosmo.h^2
    return _D_f_z(z, Ωcb0, w0wacosmo.h; mν=w0wacosmo.mν, w0=w0wacosmo.w0, wa=w0wacosmo.wa)
end
