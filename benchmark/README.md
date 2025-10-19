# Effort.jl Benchmarks

This folder contains benchmarks for Effort.jl using [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) and [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).

## Structure

- `benchmarks.jl` - Main benchmark suite definition
- `Project.toml` - Dependencies for running benchmarks
- `Manifest.toml` - Locked dependency versions (generated after first run)

## Running Benchmarks Locally

### Quick Test
To test that benchmarks are working:
```julia
julia --project=benchmark
using Pkg; Pkg.instantiate()
include("benchmark/benchmarks.jl")
using BenchmarkTools

# Run specific benchmark
run(SUITE["emulator"]["monopole"])

# Run all benchmarks
results = run(SUITE)
```

### Using AirspeedVelocity.jl
For comparing performance across commits:
```bash
# Install AirspeedVelocity
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'

# Run benchmarks on current commit
julia -e 'using AirspeedVelocity; run_asv()'

# Compare with another branch
julia -e 'using AirspeedVelocity; run_asv(ref="main")'
```

## Benchmark Categories

The benchmark suite is organized into the following groups:

### 1. Emulator Benchmarks (`SUITE["emulator"]`)
- `monopole`: Running the monopole (ℓ=0) emulator
- `quadrupole`: Running the quadrupole (ℓ=2) emulator
- `hexadecapole`: Running the hexadecapole (ℓ=4) emulator
- `all_multipoles`: Running all three multipoles together

### 2. Projection Benchmarks (`SUITE["projection"]`)
- `legendre_0/2/4`: Legendre polynomial evaluation
- `legendre_array`: Vectorized Legendre polynomial evaluation
- `apply_AP`: Fast Alcock-Paczynski effect application
- `apply_AP_check`: Numerical integration version (slower, for validation)

### 3. Integration Benchmarks (`SUITE["integration"]`)
- `window_4D`: 4D window function convolution
- `window_2D`: 2D window function convolution

### 4. Background Cosmology Benchmarks (`SUITE["background"]`)
- `E_z`: Hubble parameter evaluation
- `D_z`: Growth factor evaluation
- `f_z`: Growth rate evaluation
- `D_f_z`: Combined growth factor and rate
- `q_par_perp`: AP parameter calculation

### 5. Gradient Benchmarks (`SUITE["gradients"]`)
- `multipole_gradient`: Zygote gradient of multipole evaluation

## GitHub Actions

The benchmarks are automatically run on:
- Pull requests (comparing against base branch)
- Pushes to main/master branch (establishing baseline)

Results are posted as comments on pull requests showing performance changes.

## Adding New Benchmarks

To add a new benchmark:
1. Edit `benchmarks.jl`
2. Add to appropriate group or create new group:
```julia
SUITE["mygroup"]["mybenchmark"] = @benchmarkable my_function(x) setup = (
    x = setup_data()
)
```
3. Test locally before committing
4. Update this README if adding new categories