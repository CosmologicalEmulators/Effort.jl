using Test
using Aqua
using Effort

@testset "Aqua.jl Package Quality Assurance" begin
    # Test all Aqua checks
    #
    # We disable a few checks for now:
    # - ambiguities: often caused by upstream dependencies
    # - unbound_args: often has false positives on older Julia versions
    # - piracies: currently we have some deliberate piracies in src/chainrules.jl (for gausslobatto/LinRange)
    # - persistent_tasks: sometimes triggered by background threads from dependencies (e.g. multithreading runtimes)
    Aqua.test_all(
        Effort;
        ambiguities=false,
        unbound_args=false,
        stale_deps=true,
        deps_compat=true,
        project_extras=true,
        piracies=false,
        persistent_tasks=false
    )
end