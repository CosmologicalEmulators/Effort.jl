```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using BenchmarkTools
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/effort_benchmark.json")
```

# Effort.jl



`Effort` is a Julia package designed to emulate the computation of the Effective Field Theory of Large Scale Structure, as computed by [PyBird](https://github.com/pierrexyz/pybird). An emulator is a surrogate model, a computational technique that can mimick the behaviour of computationally expensive functions, with a speedup of several orders of magnitude.
In order to use `Effort` you need a trained emulator (after the first release, we will make them available on Zenodo). There are two different categories of trained emulators:
- single component emulators (e.g.  $P_{11}$, $P_\mathrm{loop}$, $P_\mathrm{ct}$)
- complete emulators, containing all the three different component emulators
Effort can be used as follows
```julia
import Effort
Pct_comp_array = Effort.compute_component(input_test, Pct_emu) #compute the components of Pct without the bias
Pct_array_Effort = Array{Float64}(zeros(length(Pct_comp_array[1,:]))) #allocate final array
Effort.bias_multiplication!(Pct_array_Effort, bct, Pct_comp_array) #components multiplied by bias
Effort.ComputePℓ(input_test, bs, f, Pℓ_emu) # whole multipole computation
```

This computation is quite fast: a benchmark performed locally, with a 12th Gen Intel® Core™ i7-1260P, gives the following result
```@example tutorial
benchmark[1]["Effort"]["Monopole"] # hide
```

The result of this computation look like this
![effort](https://user-images.githubusercontent.com/58727599/209453056-a83dfd18-03c2-46be-a3a5-01b5f3bd459d.png)

### Authors

- Marco Bonici, INAF - Institute of Space Astrophysics and Cosmic Physics (IASF), Milano
- Guido D'Amico, Università Degli Studi di Parma


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### License

Effort is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/CosmologicalEmulators/Effort.jl/blob/main/LICENSE) for
the full license text.
