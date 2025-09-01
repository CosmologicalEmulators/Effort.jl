using NPZ
using SimpleChains
using Static
using ForwardDiff
using Zygote
using LegendrePolynomials
using FiniteDifferences
using SciMLSensitivity
using DataInterpolations

mlpd = SimpleChain(
    static(6),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(identity, 40)
)

k_test = Array(LinRange(0, 200, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6, 2)
outminmax = rand(40, 2)
a, Ωcb0, mν, h, w0, wa = [1.0, 0.3, 0.06, 0.67, -1.1, 0.2]
z = Array(LinRange(0.0, 3.0, 100))

emu = Effort.SimpleChainsEmulator(Architecture=mlpd, Weights=weights)

postprocessing = (input, output, D, Pkemu) -> output

effort_emu = Effort.P11Emulator(TrainedEmulator=emu, kgrid=k_test, InMinMax=inminmax,
    OutMinMax=outminmax, Postprocessing=postprocessing)

x = [Ωcb0, h, mν, w0, wa]

n = 64
x1 = vcat([0.0], sort(rand(n - 2)), [1.0])
x2 = 2 .* vcat([0.0], sort(rand(n - 2)), [1.0])
y = rand(n)

W = rand(2, 20, 3, 10)
v = rand(20, 10)

function di_spline(y, x, xn)
    spline = QuadraticSpline(y, x; extrapolation=ExtrapolationType.Extension)
    return spline.(xn)
end

function D_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._D_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._f_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

function r_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._r_z(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

function r_z_check_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Effort._r_z_check(z, Ωcb0, h; mν=mν, w0=w0, wa=wa))
end

myx = Array(LinRange(0.0, 1.0, 100))
monotest = sin.(myx)
quadtest = 0.5 .* cos.(myx)
hexatest = 0.1 .* cos.(2 .* myx)
q_par = 1.4
q_perp = 0.6

x3 = Array(LinRange(-1.0, 1.0, 100))

mycosmo = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.636, ωb=0.02237, ωc=0.1, mν=0.06, w0=-2.0, wa=1.0)

run(`wget https://zenodo.org/api/records/15244205/files-archive`)
run(`unzip files-archive`)
k = npzread("k.npy")
k_test_data = npzread("k_test.npy")
Pℓ = npzread("no_AP.npy")
Pℓ_AP = npzread("yes_AP.npy")

for file in ("k.npy", "k_test.npy", "no_AP.npy", "yes_AP.npy", "files-archive")
    isfile(file) && rm(file)
end
