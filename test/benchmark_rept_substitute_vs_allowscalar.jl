using BenchmarkTools, ForwardDiff, Reactant, Enzyme, Effort, AbstractCosmologicalEmulators

sec = parse(Float64, get(ENV, "EFFORT_BENCH_SECONDS", "8.0"))
samples = parse(Int, get(ENV, "EFFORT_BENCH_SAMPLES", "80"))
BenchmarkTools.DEFAULT_PARAMETERS.seconds = sec
BenchmarkTools.DEFAULT_PARAMETERS.samples = samples
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

rept_bias(b) = begin
    b1=b[1:1]; b2=b[2:2]; bs=b[3:3]; b3=b[4:4]; a0=b[5:5]; a2=b[6:6]; a4=b[7:7]; a6=b[8:8]; sn=b[9:9]; sn2=b[10:10]; sn4=b[11:11]
    onev = b1 .* 0 .+ 1
    vcat(onev,b1,b1.*b1,b2,b1.*b2,b2.*b2,bs,b1.*bs,b2.*bs,bs.*bs,b3,b1.*b3,a0,a2,a4,a6,sn,sn2,sn4)
end
st0(k)=begin z=k.*0; k2=k.*k; k4=k2.*k2; hcat(z.+1,k2./3,k4./5) end
st2(k)=begin z=k.*0; k2=k.*k; k4=k2.*k2; hcat(z,2 .*k2./3,4 .*k4./7) end
st4(k)=begin z=k.*0; k2=k.*k; k4=k2.*k2; hcat(z,z,8 .*k4./35) end
post_lin(i,o,D,pk)=begin a=exp.(i[2:2]).*1e-10.*(D*D); o.*a end
post_loop(i,o,D,pk)=begin a=exp.(i[2:2]).*1e-10.*(D*D); o.*(a.*a) end

recomp(c,pf)=Effort.ComponentEmulator(TrainedEmulator=c.TrainedEmulator,kgrid=c.kgrid,InMinMax=c.InMinMax,OutMinMax=c.OutMinMax,Postprocessing=pf)
function rewrite_emu(e, ell)
    sm = ell==0 ? st0 : (ell==2 ? st2 : st4)
    Effort.PℓEmulator(P11=recomp(e.P11,post_lin),Ploop=recomp(e.Ploop,post_loop),Pct=recomp(e.Pct,post_lin),StochModel=sm,BiasCombination=rept_bias,JacobianBiasCombination=e.JacobianBiasCombination)
end

function get_component_traced(cos,D,comp,allow)
    ni = Effort.maximin(cos, comp.InMinMax)
    no = Effort.run_emulator(ni, comp.TrainedEmulator)
    out = Effort.inv_maximin(no, comp.OutMinMax)
    pp = allow ? (Reactant.@allowscalar comp.Postprocessing(cos,out,D,comp)) : comp.Postprocessing(cos,out,D,comp)
    reshape(pp, length(comp.kgrid), :)
end
function getP(cos,D,b,e,allow)
    p11 = get_component_traced(cos,D,e.P11,allow)
    pl  = get_component_traced(cos,D,e.Ploop,allow)
    pct = get_component_traced(cos,D,e.Pct,allow)
    st  = allow ? (Reactant.@allowscalar e.StochModel(e.P11.kgrid)) : e.StochModel(e.P11.kgrid)
    sb  = hcat(p11,pl,pct,st)
    bc  = allow ? (Reactant.@allowscalar e.BiasCombination(b)) : e.BiasCombination(b)
    vec(sum(sb .* reshape(bc, 1, :), dims=2))
end

function loss_traced(c,b,D,e0,e2,e4,allow)
    P0 = getP(c,D,b,e0,allow); P2 = getP(c,D,b,e2,allow); P4 = getP(c,D,b,e4,allow)
    sum(P0) + sum(P2) + sum(P4)
end
loss_host(c,b,D,e0,e2,e4)=begin
    P0=Effort.get_Pℓ(c,D,b,e0); P2=Effort.get_Pℓ(c,D,b,e2); P4=Effort.get_Pℓ(c,D,b,e4)
    sum(P0)+sum(P2)+sum(P4)
end

rept = Effort.trained_emulators["VelocileptorsREPTmnuw0wacdm"]
e0b,e2b,e4b = rept["0"], rept["2"], rept["4"]
e0n,e2n,e4n = rewrite_emu(e0b,0), rewrite_emu(e2b,2), rewrite_emu(e4b,4)

cosmology = [1.2,3.044,0.9649,67.36,0.02237,0.12,0.06,-1.0,0.0]
bias = [1.5,0.5,0.1,0.2,0.01,0.02,0.03,0.04,1.0,2.0,3.0]
D=0.8; ki=vec(e0b.P11.kgrid); ko=copy(ki); qp=1.02; qt=0.98; W=reshape(Base.cos.(range(0.0,3.0,length=40*length(ko))),40,length(ko))

lhb = loss_host(cosmology,bias,D,e0b,e2b,e4b)
lhn = loss_host(cosmology,bias,D,e0n,e2n,e4n)
fdcb = ForwardDiff.gradient(c->loss_host(c,bias,D,e0b,e2b,e4b), cosmology)
fdcn = ForwardDiff.gradient(c->loss_host(c,bias,D,e0n,e2n,e4n), cosmology)
fdbb = ForwardDiff.gradient(b->loss_host(cosmology,b,D,e0b,e2b,e4b), bias)
fdbn = ForwardDiff.gradient(b->loss_host(cosmology,b,D,e0n,e2n,e4n), bias)

Reactant.set_default_backend("cpu")
e0bd,e2bd,e4bd = AbstractCosmologicalEmulators.to_reactant(e0b), AbstractCosmologicalEmulators.to_reactant(e2b), AbstractCosmologicalEmulators.to_reactant(e4b)
e0nd,e2nd,e4nd = AbstractCosmologicalEmulators.to_reactant(e0n), AbstractCosmologicalEmulators.to_reactant(e2n), AbstractCosmologicalEmulators.to_reactant(e4n)
cosR,biasR = Reactant.to_rarray(cosmology), Reactant.to_rarray(bias)

lb(c,b,D,e0,e2,e4)=loss_traced(c,b,D,e0,e2,e4,true)
ln(c,b,D,e0,e2,e4)=loss_traced(c,b,D,e0,e2,e4,false)
gcb(c,b,D,e0,e2,e4)=Enzyme.gradient(Reverse,lb,c,Const(b),Const(D),Const(e0),Const(e2),Const(e4))[1]
gcn(c,b,D,e0,e2,e4)=Enzyme.gradient(Reverse,ln,c,Const(b),Const(D),Const(e0),Const(e2),Const(e4))[1]
gbb(c,b,D,e0,e2,e4)=Enzyme.gradient(Reverse,lb,Const(c),b,Const(D),Const(e0),Const(e2),Const(e4))[2]
gbn(c,b,D,e0,e2,e4)=Enzyme.gradient(Reverse,ln,Const(c),b,Const(D),Const(e0),Const(e2),Const(e4))[2]

cb = nothing
t_cb = @elapsed begin
    global cb = Reactant.@compile sync=true lb(cosR,biasR,D,e0bd,e2bd,e4bd)
    Reactant.synchronize(cb(cosR,biasR,D,e0bd,e2bd,e4bd))
end
cn = nothing
t_cn = @elapsed begin
    global cn = Reactant.@compile sync=true ln(cosR,biasR,D,e0nd,e2nd,e4nd)
    Reactant.synchronize(cn(cosR,biasR,D,e0nd,e2nd,e4nd))
end
cbgc = nothing
t_cbgc = @elapsed begin
    global cbgc = Reactant.@compile sync=true gcb(cosR,biasR,D,e0bd,e2bd,e4bd)
    Reactant.synchronize(cbgc(cosR,biasR,D,e0bd,e2bd,e4bd))
end
cngc = nothing
t_cngc = @elapsed begin
    global cngc = Reactant.@compile sync=true gcn(cosR,biasR,D,e0nd,e2nd,e4nd)
    Reactant.synchronize(cngc(cosR,biasR,D,e0nd,e2nd,e4nd))
end
cbgb = nothing
t_cbgb = @elapsed begin
    global cbgb = Reactant.@compile sync=true gbb(cosR,biasR,D,e0bd,e2bd,e4bd)
    Reactant.synchronize(cbgb(cosR,biasR,D,e0bd,e2bd,e4bd))
end
cngb = nothing
t_cngb = @elapsed begin
    global cngb = Reactant.@compile sync=true gbn(cosR,biasR,D,e0nd,e2nd,e4nd)
    Reactant.synchronize(cngb(cosR,biasR,D,e0nd,e2nd,e4nd))
end

mb(t)=BenchmarkTools.median(t).time/1e6

tb = @benchmark begin y=$cb($cosR,$biasR,$D,$e0bd,$e2bd,$e4bd); Reactant.synchronize(y); end
tn = @benchmark begin y=$cn($cosR,$biasR,$D,$e0nd,$e2nd,$e4nd); Reactant.synchronize(y); end
tbc = @benchmark begin g=$cbgc($cosR,$biasR,$D,$e0bd,$e2bd,$e4bd); Reactant.synchronize(g); end
tnc = @benchmark begin g=$cngc($cosR,$biasR,$D,$e0nd,$e2nd,$e4nd); Reactant.synchronize(g); end
tbb = @benchmark begin g=$cbgb($cosR,$biasR,$D,$e0bd,$e2bd,$e4bd); Reactant.synchronize(g); end
tnb = @benchmark begin g=$cngb($cosR,$biasR,$D,$e0nd,$e2nd,$e4nd); Reactant.synchronize(g); end

gb_cos = Array(cbgc(cosR,biasR,D,e0bd,e2bd,e4bd))
gn_cos = Array(cngc(cosR,biasR,D,e0nd,e2nd,e4nd))
gb_bias = Array(cbgb(cosR,biasR,D,e0bd,e2bd,e4bd))
gn_bias = Array(cngb(cosR,biasR,D,e0nd,e2nd,e4nd))

println("\n=== REPT substitute functions vs allowscalar baseline ===")
println("settings: seconds=$(sec), samples=$(samples)")
println("Host |Δloss| baseline vs rewritten: ", abs(lhb-lhn))
println("Host max |Δgrad cos|: ", maximum(abs.(fdcb .- fdcn)))
println("Host max |Δgrad bias|: ", maximum(abs.(fdbb .- fdbn)))
println("Reactant compile+differentiate status: OK (both paths compiled)")
println("Compile+first primal baseline(allowscalar): ", round(1000*t_cb, digits=2), " ms")
println("Compile+first primal rewritten(no allowscalar): ", round(1000*t_cn, digits=2), " ms")
println("Compile+first grad(cos) baseline(allowscalar): ", round(1000*t_cbgc, digits=2), " ms")
println("Compile+first grad(cos) rewritten(no allowscalar): ", round(1000*t_cngc, digits=2), " ms")
println("Compile+first grad(bias) baseline(allowscalar): ", round(1000*t_cbgb, digits=2), " ms")
println("Compile+first grad(bias) rewritten(no allowscalar): ", round(1000*t_cngb, digits=2), " ms")
println("Steady primal baseline(allowscalar): ", round(mb(tb), digits=4), " ms")
println("Steady primal rewritten(no allowscalar): ", round(mb(tn), digits=4), " ms")
println("Steady grad(cos) baseline(allowscalar): ", round(mb(tbc), digits=4), " ms")
println("Steady grad(cos) rewritten(no allowscalar): ", round(mb(tnc), digits=4), " ms")
println("Steady grad(bias) baseline(allowscalar): ", round(mb(tbb), digits=4), " ms")
println("Steady grad(bias) rewritten(no allowscalar): ", round(mb(tnb), digits=4), " ms")
println("Reactant max |Δgrad cos| baseline vs rewritten: ", maximum(abs.(gb_cos .- gn_cos)))
println("Reactant max |Δgrad bias| baseline vs rewritten: ", maximum(abs.(gb_bias .- gn_bias)))
println("Speedup primal rewritten/baseline: ", round(mb(tb)/mb(tn), digits=3), "x")
println("Speedup grad(cos) rewritten/baseline: ", round(mb(tbc)/mb(tnc), digits=3), "x")
println("Speedup grad(bias) rewritten/baseline: ", round(mb(tbb)/mb(tnb), digits=3), "x")
