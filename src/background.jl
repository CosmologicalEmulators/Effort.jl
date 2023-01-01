function _E_z(z, ΩM, w0, wa)
    return sqrt(ΩM*(1+z)^3+(1-ΩM)*(1+z)^(3*(1+w0+wa))*exp(-3*wa*z/(1+z)))
end

function _H_z(z, H0, ΩM, w0, wa)
    return H0*_E_z(z, ΩM, w0, wa)
end

function _r̃_z(z, ΩM, w0, wa)
    integral, _ = quadgk(x -> 1 / _E_z(x, ΩM, w0, wa), 0, z, rtol=1e-10)
    return integral
end

function _d̃A_z(z, ΩM, w0, wa)
    return (1+z) * _r̃_z(z, ΩM, w0, wa)
end

function _ΩM_z(z, ΩM, w0, wa)
    return ΩM*((1+z)^3)/(_E_z(z, ΩM, w0, wa)^2)
end

function _ρDE_z(z, w0, wa)
    return (1+z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z /(1+z))
end

function _ΩDE_z(z, ΩM, w0, wa)
    return (1-ΩM)*_ρDE_z(z, w0, wa)/(_E_z(z, ΩM, w0, wa)^2)
end

function _X_z(z, ΩM, w0, wa)
    #return _ΩM_z(z, ΩM, w0, wa)/_ΩDE_z(z, ΩM, w0, wa)
    return ΩM*((1+z)^3)/((1-ΩM)*_ρDE_z(z, w0, wa))
end

function _w_z(z, w0, wa)
    return w0+wa*z/(1+z)
end

function _growth!(du,u,p,a)
    ΩM = p[1]
    w0 = p[2]
    wa = p[3]
    z = 1.0 / a - 1.0
    G = u[1]
    dG = u[2]
    du[1] = dG
    du[2] = -(3.5-1.5*_w_z(z, w0, wa)/(1+_X_z(z, ΩM, w0, wa)))*dG/a-1.5*(1-_w_z(z, w0,wa))/(1+_X_z(z, ΩM, w0, wa))*G/(a^2)
end

function _a_z(z)
    return 1/(1+z)
end

function growth_solver(ΩM,w0,wa)
    u₀ = [1.0,0.0]

    aspan = (0.99e-3, 1.01)

    p = [ΩM, w0, wa]

    prob = ODEProblem(_growth!, u₀, aspan, p)

    sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6)
    return sol
end

function _D_z(z::Array, sol::SciMLBase.ODESolution)
    [u for (u,t) in sol.(_a_z.(z))] .* _a_z.(z) ./ (sol(_a_z(0.))[1,:])
end

function _D_z(z, sol::SciMLBase.ODESolution)
    return (Effort._a_z(z) .* sol(Effort._a_z(z))[1,:]/sol(Effort._a_z(0.))[1,:])[1,1]
end

function _f_a(a, sol::SciMLBase.ODESolution)
    G, G_prime = sol(a)
    D = G * a
    D_prime = G_prime * a + G
    return a / D * D_prime
end

function _f_z(z, sol::SciMLBase.ODESolution)
    a = _a_z(z)
    G, G_prime = sol(a)
    D = G * a
    D_prime = G_prime * a + G
    return a / D * D_prime
end
