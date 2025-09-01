using Test
using Effort

# Download test data once at the start
if !isfile("k.npy")
    println("Downloading test data from Zenodo...")
    try
        run(`wget https://zenodo.org/api/records/15244205/files-archive`)
        run(`unzip files-archive`)
    catch e
        @warn "Failed to download test data: $e"
        @warn "Some tests may fail without test data"
    end
end

include("test_helpers.jl")

@testset "Effort.jl tests" begin
    include("test_background.jl")
    include("test_akima.jl") 
    include("test_gradients.jl")
    include("test_validation.jl")
    include("test_projection.jl")
    include("test_utils.jl")
end

# Clean up test data once at the end
for file in ("k.npy", "k_test.npy", "no_AP.npy", "yes_AP.npy", "files-archive")
    isfile(file) && rm(file)
end
