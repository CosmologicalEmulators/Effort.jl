using Effort, Test

@testset "Isotropic AP remapping across the knot boundaries" begin
    k = collect(range(0.05, 0.25; length=15))
    slopes, offsets = (2.0, -1.0, 0.3), (1.5, 0.1, -0.4)
    spectra = map((a,b) -> a .* k .+ b, slopes, offsets)
    # Equal dilation factors preserve angles: Pℓ(obs, k) = Pℓ(k/q) / q³.
    # q < 1 probes above the last knot; q > 1 probes below the first.
    for q in (0.96, 1.04), method in (Effort.Akima(), Effort.Cubic())
        expected = map((a,b) -> (a .* (k ./ q) .+ b) ./ q^3, slopes, offsets)
        ref = Effort.apply_AP_check(k, k, spectra..., q, q; method)
        fast = Effort.apply_AP(k, k, spectra..., q, q; method, n_GL_points=16)
        for i in 1:3
            @test ref[i] ≈ expected[i] rtol=1e-10 atol=1e-12
            @test fast[i] ≈ ref[i] rtol=1e-10 atol=1e-12
        end
    end
end
