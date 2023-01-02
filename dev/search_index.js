var documenterSearchIndex = {"docs":
[{"location":"#Effort.jl","page":"Home","title":"Effort.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Effort is a Julia package designed to emulate the computation of the Effective Field Theory of Large Scale Structure, as computed by PyBird. An emulator is a surrogate model, a computational technique that can mimick the behaviour of computationally expensive functions, with a speedup of several orders of magnitude.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The example page shows how to use Effort.jl, while showing its computational performance","category":"page"},{"location":"#Authors","page":"Home","title":"Authors","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Marco Bonici, INAF - Institute of Space Astrophysics and Cosmic Physics (IASF), Milano\nGuido D'Amico, Università Degli Studi di Parma","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Please make sure to update tests as appropriate.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Effort is licensed under the MIT \"Expat\" license; see LICENSE for the full license text.","category":"page"},{"location":"example/#Example","page":"Example","title":"Example","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"using Plots; gr()\nPlots.reset_defaults()\nusing BenchmarkTools\ndefault(palette = palette(:tab10))\nbenchmark = BenchmarkTools.load(\"./assets/effort_benchmark.json\")","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"In order to use Effort.jl you need a trained emulator. There are two different categories of trained emulators:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"single component emulators (e.g.  P_11, P_mathrmloop, P_mathrmct)\ncomplete emulators, containing all the three different component emulators","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"In this section we are going to show how to:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"obtain a multipole power spectrum, using a trained emulator\napply the Alcock-Paczyński effect\ncompute stochastic term contribution","category":"page"},{"location":"example/#Basic-usage","page":"Example","title":"Basic usage","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Let us show how to use Effort.jl to compute Power Spectrum Multipoles.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"First of all, we need some trained emulators, then we can use the Effort.get_Pℓ function","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Effort.get_Pℓ","category":"page"},{"location":"example/#Effort.get_Pℓ","page":"Example","title":"Effort.get_Pℓ","text":"get_Pℓ(cosmology::Array, bs::Array, f, cosmoemu::AbstractPℓEmulators)\n\nCompute the Pℓ array given the cosmological parameters array cosmology, the bias array bs, the growth factor f and an AbstractEmulator.\n\n\n\n\n\n","category":"function"},{"location":"example/","page":"Example","title":"Example","text":"info: Trained emulators\nRight now we do not provide any emulator, but with the paper publication we will release several trained emulators on Zenodo.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"import Effort\nPct_comp_array = Effort.compute_component(input_test, Pct_Mono_emu) #compute the components of Pct without the bias\nPct_array_Effort = Array{Float64}(zeros(length(Pct_comp_array[1,:]))) #allocate final array\nEffort.bias_multiplication!(Pct_array_Effort, bct, Pct_comp_array) #components multiplied by bias\nEffort.get_Pℓ(input_test, bs, f, Pℓ_Mono_emu) # whole multipole computation","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Here we are using a ComponentEmulator, which can compute one of the components as predicted by PyBird, and a MultipoleEmulator, which emulates an entire multipole.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"This computation is quite fast: a benchmark performed locally, with a 12th Gen Intel® Core™ i7-1260P, gives the following result for a multipole computation","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"benchmark[1][\"Effort\"][\"Monopole\"] # hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The result of these computations look like this (Image: effort)","category":"page"},{"location":"example/#Alcock-Paczyński-effect","page":"Example","title":"Alcock-Paczyński effect","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Here we are going to write down the equations related to the AP effect, following the Ivanov et al. (2019) and D'Amico et al. (2020) notation.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"In particular, we are going to use:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"rmref, for the quantities evaluated in the reference cosmology used to perform the measurements\nrmtrue, for the quantities evaluated in the true cosmology used to perform the theoretical predictions\nmathrmobs, for the observed quantities after applying the AP effect","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The wavenumbers parallel and perpendicular to the line of sight (k^mathrmtrue_parallel k^mathrmtrue_perp) are related to the ones of the reference cosmology as (k^mathrmref_parallel k^mathrmref_perp) as:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"k_^text ref =q_ k^mathrmtrue_ quad k_perp^mathrmref=q_perp k^mathrmtrue_perp","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"where the distortion parameters are defined by","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"q_=fracD^mathrmtrue_A(z) H^mathrmtrue(z=0)D_A^mathrmref(z) H^mathrmref(z=0) quad q_perp=fracH^mathrmref(z)  H^mathrmref(z=0)H^mathrmtrue(z)  H^mathrmtrue(z=0)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"where D^A, H are the angular diameter distance and Hubble parameter, respectively. In terms of these parameters, the power spectrum multipoles in the reference cosmology is given by the multipole projection integral","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"P_ell mathrmAP(k)=frac2 ell+12 int_-1^1 d mu_mathrmobs P_mathrmobsleft(k_mathrmobs mu_mathrmobsright) cdot mathcalP_ellleft(mu_mathrmobsright)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The observed P_mathrmobsleft(k_mathrmobs mu_mathrmobsright), when including the AP effect, is given by","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"P_mathrmobsleft(k_mathrmobs mu_mathrmobsright)= frac1q_ q_perp^2 cdot P_gleft(k_text true leftk_mathrmobs mu_mathrmobsright mu_text true leftk_text obs  mu_mathrmobsrightright)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"In the Effort.jl workflow, the Alcock-Paczyński (AP) effect can be included in two different ways:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"by training the emulators using spectra where the AP effect has already been applied\nby using standard trained emulators and applying analitycally the AP effect","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"While the former approach is computationally faster (there is no overhead from the NN point-of-view), the latter is more flexible, since the reference cosmology for the AP effect computation can be changed at runtime.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Regarding the second approach, the most important choice regards the algorithm employed to compute the multipole projection integral. Here we implement two different approaches, based on","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"QuadGK.jl. This approach is the most precise, since it uses an adaptive method to compute the integral.\nFastGaussQuadrature.jl. This approach is the fastest, since we are going to employ only 5 points to compute the integral, taking advantage of the Gauss-Lobatto quadrature rule.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"In order to understand why it is possible to use few points to evaluate the AP projection integral, it is intructive to plot the mu dependence of the integrand","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: mu_dependence)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The ell=4 integrand, the most complicated one, can be accurately fit with a n=8 polynomial","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: polyfit_residuals)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Since a n Gauss-Lobatto rule can integrate exactly 2n  3 degree polynomials,  we expect that a GL rule with 10 points can perform the integral with high precision.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Now we can show how to use Effort.jl to compute the AP effect using the GK adaptive integration","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Effort.apply_AP_check","category":"page"},{"location":"example/#Effort.apply_AP_check","page":"Example","title":"Effort.apply_AP_check","text":"apply_AP_check(k_grid::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array,\nq_par, q_perp)\n\nGiven the Monopole, the Quadrupole, the Hexadecapole, and the k-grid, this function apply the AP effect using the Gauss-Kronrod adaptive quadrature. Precise, but expensive, function. Mainly used for check and debugging purposes.\n\n\n\n\n\n","category":"function"},{"location":"example/","page":"Example","title":"Example","text":"import Effort\nEffort.apply_AP_check(k_test, Mono_Effort, Quad_Effort, Hexa_Effort,  q_par, q_perp)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"benchmark[1][\"Effort\"][\"AP_GK\"] # hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"As said, this is precise but a bit expensive from a computational point of view. What about Gauss-Lobatto?","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Effort.apply_AP","category":"page"},{"location":"example/#Effort.apply_AP","page":"Example","title":"Effort.apply_AP","text":"apply_AP(k_grid::Array, Mono_array::Array, Quad_array::Array, Hexa_array::Array, q_par,\nq_perp)\n\nGiven the Monopole, the Quadrupole, the Hexadecapole, and the k-grid, this function apply the AP effect using the Gauss-Lobatto quadrature. Fast but accurate,  well tested against adaptive Gauss-Kronrod integration.\n\n\n\n\n\n","category":"function"},{"location":"example/","page":"Example","title":"Example","text":"import Effort\nEffort.apply_AP(k_test, Mono_Effort, Quad_Effort, Hexa_Effort,  q_par, q_perp)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"benchmark[1][\"Effort\"][\"AP_GL\"] # hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"This is ten times faster than the adaptive integration, but is also very accurate! A comparison with the GK-based rule show a percentual relative difference of about 10^-11 for the Hexadecapole, with a higher precision for the other two multipoles.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: gk_gl_residuals)","category":"page"},{"location":"example/#Growth-factor","page":"Example","title":"Growth factor","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"A quantity required to compute EFTofLSS observables is the growth rate, f. While other emulator packages employ an emulator also for f (or equivalently emulate the growth factor D), we choose a different approach, using the DiffEq.jl library to efficiently solve the equation for the growth factor, as written in Jenkins & Linder (2003)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"D^prime prime+frac32left1-fracw(a)1+X(a)right fracD^primea-frac32 fracX(a)1+X(a) fracDa^2=0","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Performing the sostitution G=Da, the previous equation becomes","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"G^prime prime+leftfrac72-frac32 fracw_DE(a)1+X(a)right fracG^primea+frac32 frac1-w_DE(a)1+X(a) fracGa^2=0","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Since we start solving the equation deep in the matter dominated era, when G(a)sim 1, we can set as initial conditions","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"G(z_i) = 1","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"G(z_i)=0","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Computation is quite fast","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"@benchmark Effort._D_z($z, $ΩM, $w0, $wa)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"benchmark[1][\"Effort\"][\"AP_GL\"] # hide","category":"page"}]
}
