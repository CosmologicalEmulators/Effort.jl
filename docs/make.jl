using Documenter
using Plots
using Effort


ENV["GKSwstype"] = "100"

push!(LOAD_PATH,"../src/")

makedocs(
    modules = [Effort],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
    sidebar_sitename=true),
    sitename = "Effort.jl",
    authors  = "Marco Bonici",
    pages = [
        "Home" => "index.md"
        "Example" => "example.md"
    ]
)

deploydocs(
    repo = "github.com/CosmologicalEmulators/Effort.jl.git",
    devbranch = "develop"
)
