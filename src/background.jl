function E_z(z, ΩM, w0, wa)
    return sqrt(ΩM*(1+z)^3+(1-ΩM)*(1+z)^(3*(1+w0+wa))*exp(-3*wa*z/(1+z)))
end

function H_z(z, H0, ΩM, w0, wa)
    return H0*E_z(z, ΩM, w0, wa)
end

function r̃_z(z, ΩM, w0, wa)
    integral, _ = quadgk(x -> 1 / E_z(x, ΩM, w0, wa), 0, z, rtol=1e-10)
    return integral
end

function d̃A_z(z, ΩM, w0, wa)
    return (1+z) * r̃_z(z, ΩM, w0, wa)
end
