#TODO: this part should be moved to a dedicate package. While necessary to a full Effort
#functionality, this could be factorized to a new module, specifically taylored to this goal
# and maybe used in other packages

function _F(y)
    result, _ = quadgk(x -> x^2*√(x^2+y^2)/(1+exp(x)), 0, Inf, rtol=1e-12)
    return result
end

function get_y(mν, a; kB = 8.617342e-5, Tν = 0.71611*2.7255)
    return mν * a / (kB * Tν)
end

function _dFdy(y)
    result, _ = quadgk(x -> x^2/((1+exp(x))*√(x^2+y^2)), 0, Inf, rtol=1e-12)
    return result*y
end

function _ΩνE2(a, Ωγ0, mν; kB = 8.617342e-5, Tν = 0.71611*2.7255, Neff = 3.044)
    Γν = (4/11)^(1/3)*(Neff/3)^(1/4)#0.71649^4
    return 15/π^4*Γν^4*Ωγ0/a^4*F_interpolant(get_y(mν, a))
end

function _ΩνE2(a, Ωγ0, mν::Array; kB = 8.617342e-5, Tν = 0.71611*2.7255, Neff = 3.044)
    Γν = (4/11)^(1/3)*(Neff/3)^(1/4)#0.71649^4
    sum_interpolant = 0.
    for mymν in mν
        sum_interpolant += F_interpolant(get_y(mymν, a))
    end
    return 15/π^4*Γν^4*Ωγ0/a^4*sum_interpolant
end

function _dΩνE2da(a, Ωγ0, mν; kB = 8.617342e-5, Tν = 0.71611*2.7255, Neff = 3.044)
    Γν = (4/11)^(1/3)*(Neff/3)^(1/4)#0.71649^4
    return 15/π^4*Γν^4*Ωγ0*(-4*F_interpolant(get_y(mν, a))/a^5+
           dFdy_interpolant(get_y(mν, a))/a^4*(mν/kB/Tν))
end

function _dΩνE2da(a, Ωγ0, mν::Array; kB = 8.617342e-5, Tν = 0.71611*2.7255, Neff = 3.044)
    Γν = (4/11)^(1/3)*(Neff/3)^(1/4)#0.71649^4
    sum_interpolant = 0.
    for mymν in mν
        sum_interpolant += -4*F_interpolant(get_y(mymν, a))/a^5+
        dFdy_interpolant(get_y(mymν, a))/a^4*(mymν/kB/Tν)
    end
    return 15/π^4*Γν^4*Ωγ0*sum_interpolant
end

"""function _E_z(z, ΩM, w0, wa)
    return sqrt(ΩM*(1+z)^3+(1-ΩM)*_ρDE_z(z, w0, wa))
end
"""
function _E_a(a, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    Ωγ0 = 2.469e-5/h^2
    Ων0 = _ΩνE2(1., Ωγ0, mν)
    ΩΛ0 = 1. - (Ωγ0 + Ωc0 + Ωb0 + Ων0)
    return sqrt(Ωγ0*a^-4 + (Ωc0 + Ωb0)*a^-3 + ΩΛ0 * _ρDE_a(a, w0, wa)+ _ΩνE2(a, Ωγ0, mν))
end

"""function _H_z(z, H0, ΩM, w0, wa)
    return H0*_E_z(z, ΩM, w0, wa)
end"""

_H_a(a, Ωγ0, Ωc0, Ωb0, mν, h, w0, wa) = 100*h*_E_a(a, Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa)

function _χ_z(z, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    integral, _ = quadgk(x -> 1 /
    _E_a(_a_z(x), Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa), 0, z, rtol=1e-6)
    return integral*c_0/(100*h)
end

function _dEda(a, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    Ωγ0 = 2.469e-5/h^2
    Ων0 = _ΩνE2(1., Ωγ0, mν)
    ΩΛ0 = 1. - (Ωγ0 + Ωc0 + Ωb0 + Ων0)
    return 0.5/_E_a(a, Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa)*(-3(Ωc0 + Ωb0)a^-4-4Ωγ0*a^-5+
           ΩΛ0*_dρDEda(a, w0, wa)+_dΩνE2da(a, Ωγ0, mν))
end

function _dlogEdloga(a, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    Ωγ0 = 2.469e-5/h^2
    Ων0 = _ΩνE2(1., Ωγ0, mν)
    ΩΛ0 = 1. - (Ωγ0 + Ωc0 + Ωb0 + Ων0)
    return a*0.5/(_E_a(a, Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa)^2)*(-3(Ωc0 + Ωb0)a^-4-4Ωγ0*a^-
           5+ΩΛ0*_dρDEda(a, w0, wa)+_dΩνE2da(a, Ωγ0, mν))
end

function _ΩMa(a, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    Ωγ0 = 2.469e-5/h^2
    Ων0 = _ΩνE2(1., Ωγ0, mν)
    return (Ωc0 + Ωb0 + Ων0 )*a^-3 / (_E_a(a, Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa))^2
end

function _r̃_z(z, ΩM, w0, wa)
    integral, _ = quadgk(x -> 1 / _E_z(x, ΩM, w0, wa), 0, z, rtol=1e-10)
    return integral
end

function _r_z(z, H0, ΩM, w0, wa)
    return c_0 * _r̃_z(z, ΩM, w0, wa) / H0
end

function _d̃A_z(z, ΩM, w0, wa)
    return _r̃_z(z, ΩM, w0, wa) / (1+z)
end

function _dA_z(z, H0, ΩM, w0, wa)
    return _r_z(z, H0, ΩM, w0, wa) / (1+z)
end

function _ρDE_z(z, w0, wa)
    return (1+z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z /(1+z))
end

function _ρDE_a(a, w0, wa)
    return a^(-3.0 * (1.0 + w0 + wa)) * exp(3.0 * wa * (a-1))
end

function _dρDEda(a, w0, wa)
    return 3*(-(1+w0+wa)/a+wa)*_ρDE_a(a, w0, wa)
end

function _X_z(z, ΩM, w0, wa)
    return ΩM*((1+z)^3)/((1-ΩM)*_ρDE_z(z, w0, wa))
end

function _w_z(z, w0, wa)
    return w0+wa*z/(1+z)
end

"""function _growth!(du,u,p,a)
    ΩM = p[1]
    w0 = p[2]
    wa = p[3]
    z = 1.0 / a - 1.0
    G = u[1]
    dG = u[2]
    du[1] = dG
    du[2] = -(3.5-1.5*_w_z(z, w0, wa)/(1+_X_z(z, ΩM, w0, wa)))*dG/a-1.5*(1-_w_z(z, w0,wa))/(1+_X_z(z, ΩM, w0, wa))*G/(a^2)
end"""

function _growth!(du,u,p,loga)
    #Ωγ0 = p[1]
    Ωc0 = p[1]
    Ωb0 = p[2]
    mν  = p[3]
    h   = p[4]
    w0  = p[5]
    wa  = p[6]
    a = exp(loga)
    D = u[1]
    dD = u[2]
    du[1] = dD
    du[2] = -(2+_dlogEdloga(a, Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa))*dD+
            1.5*_ΩMa(a, Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa)*D
end

function _a_z(z)
    return 1/(1+z)
end

"""function growth_solver(ΩM, w0, wa)
    u₀ = [1.0,0.0]

    aspan = (0.99e-3, 1.01)

    p = [ΩM, w0, wa]

    prob = ODEProblem(_growth!, u₀, aspan, p)

    sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6;verbose=false)
    return sol
end"""

function growth_solver(Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    amin = 1/139
    u₀ = [amin, amin]

    logaspan = (log(amin), log(1.01))
    Ωγ0 = 2.469e-5/h^2

    p = (Ωc0, Ωb0 , mν, h, w0, wa)

    prob = ODEProblem(_growth!, u₀, logaspan, p)

    sol = solve(prob, OrdinaryDiffEq.Tsit5(), abstol=1e-6, reltol=1e-6; verbose=false)
    return sol
end

"""function _D_z(z::Array, sol::SciMLBase.ODESolution)
    [u for (u,t) in sol.(_a_z.(z))] .* _a_z.(z) ./ (sol(_a_z(0.))[1,:])
end

function _D_z(z, sol::SciMLBase.ODESolution)
    return (Effort._a_z(z) .* sol(Effort._a_z(z))[1,:]/sol(Effort._a_z(0.))[1,:])[1,1]
end

function _D_z(z, ΩM, w0, wa)
    sol = growth_solver(ΩM, w0, wa)
    return _D_z(z, sol)
end

function _f_a(a, sol::SciMLBase.ODESolution)
    G, G_prime = sol(a)
    D = G * a
    D_prime = G_prime * a + G
    return a / D * D_prime
end

function _f_a(a::Array, sol::SciMLBase.ODESolution)
    G = [u for (u,t) in sol.(a)]
    G_prime = [t for (u,t) in sol.(a)]
    D = G .* a
    D_prime = G_prime .* a .+ G
    return a ./ D .* D_prime
end

function _f_z(z, sol::SciMLBase.ODESolution)
    a = _a_z.(z)
    return _f_a(a, sol)
end

function _f_z(z, ΩM, w0, wa)
    sol = growth_solver(ΩM, w0, wa)
    return _f_z(z, sol)
end"""

function _D_z(z::Array, sol::SciMLBase.ODESolution)
    [u for (u,t) in sol.(log.(_a_z.(z)))] ./ (sol(log(_a_z(0.)))[1,:])
end

function _D_z(z, sol::SciMLBase.ODESolution)
    return (sol(log(_a_z(z)))[1,:]/sol(log(_a_z(0.)))[1,:])[1,1]
end

function _D_z(z, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    sol = growth_solver(Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa)
    return _D_z(z, sol)
end

function _D_z_unnorm(z::Array, sol::SciMLBase.ODESolution)
    [u for (u,t) in sol.(log.(_a_z.(z)))]
end

function _D_z_unnorm(z, sol::SciMLBase.ODESolution)
    return (sol(log(_a_z(z)))[1,:])[1,1]
end

function _f_a(a, sol::SciMLBase.ODESolution)
    D, D_prime = sol(log(a))
    return 1 / D * D_prime
end

function _f_z(z, sol::SciMLBase.ODESolution)
    a = _a_z.(z)
    return _f_a(a, sol)
end

function _f_z(z, Ωc0, Ωb0, h; mν =0., w0=-1., wa=0.)
    a = _a_z.(z)
    sol = growth_solver(Ωc0, Ωb0, h; mν =mν, w0=w0, wa=wa)
    return _f_a(a, sol)
end
