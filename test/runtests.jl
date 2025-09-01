using Test
using Effort

include("test_helpers.jl")

@testset "Effort.jl tests" begin
    include("test_background.jl")
    include("test_akima.jl") 
    include("test_gradients.jl")
    include("test_validation.jl")
end
