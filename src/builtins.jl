ln10As_growth_linear(input, output, D, Pkemu) = begin
    ln10As = input[2:2]
    amp = exp.(ln10As) .* 1e-10 .* (D .* D)
    output .* amp
end

ln10As_growth_quadratic(input, output, D, Pkemu) = begin
    ln10As = input[2:2]
    amp = exp.(ln10As) .* 1e-10 .* (D .* D)
    output .* (amp .* amp)
end

pybird_eftoflss_bias_combination(bs) = begin
    b1 = bs[1:1]; b2 = bs[2:2]; b3 = bs[3:3]; b4 = bs[4:4]
    b5 = bs[5:5]; b6 = bs[6:6]; b7 = bs[7:7]; f = bs[8:8]
    cϵ0 = bs[9:9]; cϵ1 = bs[10:10]; cϵ2 = bs[11:11]
    onev = b1 .* 0 .+ 1
    vcat(
        b1 .* b1, 2 .* b1 .* f, f .* f, onev, b1, b2, b3, b4,
        b1 .* b1, b1 .* b2, b1 .* b3, b1 .* b4, b2 .* b2,
        b2 .* b4, b4 .* b4, 2 .* b1 .* b5, 2 .* b1 .* b6,
        2 .* b1 .* b7, 2 .* f .* b5, 2 .* f .* b6, 2 .* f .* b7,
        cϵ0, cϵ1, cϵ2 .* f,
    )
end

velocileptors_bias_combination(biases) = begin
    b1 = biases[1:1]; b2 = biases[2:2]; bs = biases[3:3]; b3 = biases[4:4]
    alpha0 = biases[5:5]; alpha2 = biases[6:6]; alpha4 = biases[7:7]; alpha6 = biases[8:8]
    sn = biases[9:9]; sn2 = biases[10:10]; sn4 = biases[11:11]
    onev = b1 .* 0 .+ 1
    vcat(
        onev, b1, b1 .* b1, b2, b1 .* b2, b2 .* b2, bs,
        b1 .* bs, b2 .* bs, bs .* bs, b3, b1 .* b3, alpha0,
        alpha2, alpha4, alpha6, sn, sn2, sn4,
    )
end

pybird_eftoflss_jacobian_bias_combination(bs) = begin
    b1 = bs[1:1]; b2 = bs[2:2]; b3 = bs[3:3]; b4 = bs[4:4]
    b5 = bs[5:5]; b6 = bs[6:6]; b7 = bs[7:7]; f = bs[8:8]
    cϵ2 = bs[11:11]
    z = b1 .* 0; o = z .+ 1
    col1 = vcat(2 .* b1, 2 .* f, z, z, o, z, z, z, 2 .* b1, b2, b3, b4, z, z, z, 2 .* b5, 2 .* b6, 2 .* b7, z, z, z, z, z, z)
    col2 = vcat(z, z, z, z, z, o, z, z, z, b1, z, z, 2 .* b2, b4, z, z, z, z, z, z, z, z, z, z)
    col3 = vcat(z, z, z, z, z, z, o, z, z, z, b1, z, z, z, z, z, z, z, z, z, z, z, z, z)
    col4 = vcat(z, z, z, z, z, z, z, o, z, z, z, b1, z, b2, 2 .* b4, z, z, z, z, z, z, z, z, z)
    col5 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2 .* b1, z, z, 2 .* f, z, z, z, z, z)
    col6 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2 .* b1, z, z, 2 .* f, z, z, z, z)
    col7 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2 .* b1, z, z, 2 .* f, z, z, z)
    col8 = vcat(z, 2 .* b1, 2 .* f, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2 .* b5, 2 .* b6, 2 .* b7, z, z, cϵ2)
    col9 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z, z)
    col10 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z)
    col11 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, f)
    hcat(col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11)
end

velocileptors_jacobian_bias_combination(biases) = begin
    b1 = biases[1:1]; b2 = biases[2:2]; bs = biases[3:3]; b3 = biases[4:4]
    z = b1 .* 0; o = z .+ 1
    col1 = vcat(z, o, 2 .* b1, z, b2, z, z, bs, z, z, z, b3, z, z, z, z, z, z, z)
    col2 = vcat(z, z, z, o, b1, 2 .* b2, z, z, bs, z, z, z, z, z, z, z, z, z, z)
    col3 = vcat(z, z, z, z, z, z, o, b1, b2, 2 .* bs, z, z, z, z, z, z, z, z, z)
    col4 = vcat(z, z, z, z, z, z, z, z, z, z, o, b1, z, z, z, z, z, z, z)
    col5 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, o, z, z, z, z, z, z)
    col6 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, o, z, z, z, z, z)
    col7 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z, z, z, z)
    col8 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z, z, z)
    col9 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z, z)
    col10 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z)
    col11 = vcat(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o)
    hcat(col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11)
end

const BUILTIN_COMPONENT_POSTPROCESSING = Dict{String,Function}(
    "ln10As_growth_linear" => ln10As_growth_linear,
    "ln10As_growth_quadratic" => ln10As_growth_quadratic,
)

const BUILTIN_STOCHMODELS = Dict{String,Function}(
    "pybird_ell0" => function(k)
        z = k .* 0
        km2 = 0.49
        k_rescaled = (k .* k) ./ km2
        hcat(z .+ 1, k_rescaled, k_rescaled ./ 3)
    end,
    "pybird_ell2" => function(k)
        z = k .* 0
        km2 = 0.49
        k_rescaled = (k .* k) ./ km2
        hcat(z, z, k_rescaled .* (2/3))
    end,
    "pybird_ell4" => function(k)
        z = k .* 0
        hcat(z, z, z)
    end,
    "velocileptors_ell0" => function(k)
        z = k .* 0
        k2 = k .* k
        k4 = k2 .* k2
        hcat(z .+ 1, k2 ./ 3, k4 ./ 5)
    end,
    "velocileptors_ell2" => function(k)
        z = k .* 0
        k2 = k .* k
        k4 = k2 .* k2
        hcat(z, 2 .* k2 ./ 3, 4 .* k4 ./ 7)
    end,
    "velocileptors_ell4" => function(k)
        z = k .* 0
        k2 = k .* k
        k4 = k2 .* k2
        hcat(z, z, 8 .* k4 ./ 35)
    end,
)

const BUILTIN_BIAS_COMBINATIONS = Dict{String,Function}(
    "pybird_eftoflss" => pybird_eftoflss_bias_combination,
    "velocileptors" => velocileptors_bias_combination,
)

const BUILTIN_JAC_BIAS_COMBINATIONS = Dict{String,Function}(
    "pybird_eftoflss" => pybird_eftoflss_jacobian_bias_combination,
    "velocileptors" => velocileptors_jacobian_bias_combination,
)
