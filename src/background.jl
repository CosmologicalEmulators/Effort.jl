#TODO: this part should be moved to a dedicate package. While necessary to a full Effort
#functionality, this could be factorized to a new module, specifically taylored to this goal
# and maybe used in other packages, maybe in AbstractCosmologicalEmulators?

function _F(y)
    f(x, y) = x^2 * √(x^2 + y^2) / (1 + exp(x))
    domain = (zero(eltype(Inf)), Inf) # (lb, ub)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol
end

function _get_y(mν, a; kB=8.617342e-5, Tν=0.71611 * 2.7255)
    return mν * a / (kB * Tν)
end

function _dFdy(y)
    f(x, y) = x^2 / ((1 + exp(x)) * √(x^2 + y^2))
    domain = (zero(eltype(Inf)), Inf) # (lb, ub)
    prob = IntegralProblem(f, domain, y; reltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol * y
end

function _ΩνE2(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * F_interpolant(_get_y(mν, a))
end

function _ΩνE2(a, Ωγ0, mν::Array; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)#0.71649^4
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += F_interpolant(_get_y(mymν, a))
    end
    return 15 / π^4 * Γν^4 * Ωγ0 / a^4 * sum_interpolant
end

function _dΩνE2da(a, Ωγ0, mν; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)#0.71649^4
    return 15 / π^4 * Γν^4 * Ωγ0 * (-4 * F_interpolant(_get_y(mν, a)) / a^5 +
                                    dFdy_interpolant(_get_y(mν, a)) / a^4 * (mν / kB / Tν))
end

function _dΩνE2da(a, Ωγ0, mν::Array; kB=8.617342e-5, Tν=0.71611 * 2.7255, Neff=3.044)
    Γν = (4 / 11)^(1 / 3) * (Neff / 3)^(1 / 4)#0.71649^4
    sum_interpolant = 0.0
    for mymν in mν
        sum_interpolant += -4 * F_interpolant(_get_y(mymν, a)) / a^5 +
                           dFdy_interpolant(_get_y(mymν, a)) / a^4 * (mymν / kB / Tν)
    end
    return 15 / π^4 * Γν^4 * Ωγ0 * sum_interpolant
end

function _E_a(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return @. sqrt(Ωγ0 * a^-4 + Ωcb0 * a^-3 + ΩΛ0 * _ρDE_a(a, w0, wa) + _ΩνE2(a, Ωγ0, mν))
end

function _E_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    a = _a_z.(z)
    return _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)
end

#TODO check whether to cancel this one
_H_a(a, Ωcb0, mν, h, w0, wa) = 100 * h * _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)

#TODO check whether to cancel this one
function _χ_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    p = [Ωcb0, h, mν, w0, wa]
    f(x, p) = 1 / _E_a(_a_z(x), p[1], p[2]; mν=p[3], w0=p[4], wa=p[5])
    domain = (zero(eltype(z)), z) # (lb, ub)
    prob = IntegralProblem(f, domain, y; preltol=1e-12)
    sol = solve(prob, QuadGKJL())[1]
    return sol * c_0 / (100 * h)
end

#TODO check whether to cancel this one
function _dEda(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return 0.5 / _E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa) *
    (-3(Ωcb0)a^-4 - 4Ωγ0 * a^-5 + ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν))
end

function _dlogEdloga(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)
    return a * 0.5 / (_E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa)^2) * (-3(Ωcb0)a^-4 - 4Ωγ0 * a^-
                                                                 5 + ΩΛ0 * _dρDEda(a, w0, wa) + _dΩνE2da(a, Ωγ0, mν))
end

function _Ωcba(a, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    Ωγ0 = 2.469e-5 / h^2
    Ων0 = _ΩνE2(1.0, Ωγ0, mν)
    return (Ωcb0 + Ων0) * a^-3 / (_E_a(a, Ωcb0, h; mν=mν, w0=w0, wa=wa))^2
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

function _r_z_check(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _r̃_z_check(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

function _r_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return c_0 * _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (100 * h)
end

function _d̃A_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return _r̃_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (1 + z)
end

function _dA_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    return _r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa) / (1 + z)
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

#TODO check whether we can remove this function
function _X_z(z, Ωcb0, w0, wa)
    return Ωcb0 * ((1 + z)^3) / ((1 - Ωcb0) * _ρDE_z(z, w0, wa))
end

function _w_z(z, w0, wa)
    return w0 + wa * z / (1 + z)
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
            1.5 * _Ωcba(a, Ωcb0, h; mν=mν, w0=w0, wa=wa) * D
end

function _a_z(z)
    return @. 1 / (1 + z)
end

function _growth_solver(Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    u₀ = [amin, amin]

    logaspan = (log(amin), log(1.01))#to ensure we cover the relevant range
    #Ωγ0 = 2.469e-5 / h^2

    p = (Ωcb0, mν, h, w0, wa)

    prob = ODEProblem(_growth!, u₀, logaspan, p)

    sol = solve(prob, OrdinaryDiffEq.Tsit5(), abstol=1e-6, reltol=1e-6; verbose=false)
    return sol
end

function _growth_solver(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    amin = 1 / 139
    loga = vcat(log.(_a_z.(z)), 0.0)
    u₀ = [amin, amin]

    logaspan = (log(amin), log(1.01))
    #Ωγ0 = 2.469e-5 / h^2

    p = [Ωcb0, mν, h, w0, wa]

    prob = ODEProblem(_growth!, u₀, logaspan, p)

    sol = solve(prob, OrdinaryDiffEq.Tsit5(), abstol=1e-6, reltol=1e-6; saveat=loga)[1:2, :]
    return sol
end

function _D_z_old(z::Array, sol::SciMLBase.ODESolution)
    [u for (u, t) in sol.(log.(_a_z.(z)))] ./ (sol(log(_a_z(0.0)))[1, :])
end

function _D_z_old(z, sol::SciMLBase.ODESolution)
    return (sol(log(_a_z(z)))[1, :]/sol(log(_a_z(0.0)))[1, :])[1, 1]
end

function _D_z(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    D_z = reverse(sol[1, 1:end-1]) ./ sol[1, end]
    return D_z
end

function _D_z_old(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return _D_z_old(z, sol)
end

function _D_z_unnorm(z, Ωcb0, h; mν=0.0, w0=-1.0, wa=0.0)
    sol = _growth_solver(Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return _D_z_unnorm(z, sol)
end

function _D_z_unnorm(z::Array, sol::SciMLBase.ODESolution)
    [u for (u, t) in sol.(log.(_a_z.(z)))]
end

function _D_z_unnorm(z, sol::SciMLBase.ODESolution)
    return (sol(log(_a_z(z)))[1, :])[1, 1]
end

function _f_a_old(a, sol::SciMLBase.ODESolution)
    D, D_prime = sol.(log.(a))
    return @. 1 / D * D_prime
end

function _f_z_old(z, sol::SciMLBase.ODESolution)
    a = _a_z.(z)
    return _f_a_old(a, sol)
end

function _f_z_old(z, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
    a = _a_z.(z)
    sol = _growth_solver(Ωcb0, h; mν=mν, w0=w0, wa=wa)
    return _f_a_old(a, sol)
end

function _f_z(z, Ωcb0, h; mν=0, w0=-1.0, wa=0.0)
    sol = _growth_solver(z, Ωcb0, h; mν=mν, w0=w0, wa=wa)
    D = sol[1, 1:end-1]
    D_prime = sol[2, 1:end-1]
    return @. 1 / D * D_prime
end
