using Documenter
using Plots
using Effort


ENV["GKSwstype"] = "100"

push!(LOAD_PATH, "../src/")

makedocs(
    modules=[Effort],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true",
        sidebar_sitename=false),
    sitename="Effort.jl",
    authors="Marco Bonici",
    pages=[
        "Home" => "index.md"
        "Example" => "example.md"
        "API Documentation" => [
                    "External API" => "api_external.md", # Link to a file that lists external API
                    "Internal API" => "api_internal.md",   # Link to a file that lists internal API
                ]
    ]
)

deploydocs(
    repo="github.com/CosmologicalEmulators/Effort.jl.git",
    devbranch="develop"
)
