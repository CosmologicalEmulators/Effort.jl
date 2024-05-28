module Effort

using Base: @kwdef
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description
using DataInterpolations
using FastGaussQuadrature
using LegendrePolynomials
using LoopVectorization
using Memoization
using OrdinaryDiffEq
using QuadGK

const c_0 = 2.99792458e5

function __init__()
    min_y = get_y(0,0) #obvious, I know
    max_y = get_y(1,10)
    y_grid = vcat(LinRange(min_y, 100, 100), LinRange(100.1, max_y, 1000))
    F_grid = [_F(y) for y in y_grid]
    global F_interpolant = AkimaInterpolation(F_grid, y_grid)
    y_grid = vcat(LinRange(min_y, 10., 10000), LinRange(10.1, max_y, 10000))
    dFdy_grid = [_dFdy(y) for y in y_grid]
    global dFdy_interpolant = AkimaInterpolation(dFdy_grid, y_grid)
end

include("background.jl")
include("neural_networks.jl")
include("eft_commands.jl")
include("projection.jl")

end # module
