"""
Tests for background cosmology functionality.

Tests cover:
- Cosmology object creation
- q_par and q_perp computation (AP parameters)
- Correctness against expected values
"""

using Test
using Effort

@testset "Background Cosmology" begin
    @testset "Cosmology Object Creation" begin
        # Test creating a cosmology object
        cosmo = Effort.w0waCDMCosmology(
            ln10Aₛ=3.0, nₛ=0.96, h=0.636,
            ωb=0.02237, ωc=0.1, mν=0.06,
            w0=-2.0, wa=1.0
        )

        @test cosmo isa Effort.w0waCDMCosmology
        @test cosmo.h ≈ 0.636
        @test cosmo.ωb ≈ 0.02237
        @test cosmo.ωc ≈ 0.1
        @test cosmo.mν ≈ 0.06
        @test cosmo.w0 ≈ -2.0
        @test cosmo.wa ≈ 1.0
    end

    @testset "q_par_perp Computation" begin
        # Test cosmology
        cosmo_test = Effort.w0waCDMCosmology(
            ln10Aₛ=3.0, nₛ=0.96, h=0.636,
            ωb=0.02237, ωc=0.1, mν=0.06,
            w0=-2.0, wa=1.0
        )

        # Reference cosmology
        cosmo_ref = Effort.w0waCDMCosmology(
            ln10Aₛ=3.0, nₛ=0.96, h=0.6736,
            ωb=0.02237, ωc=0.12, mν=0.06,
            w0=-1.0, wa=0.0
        )

        # Compute q_par and q_perp at z=0.5
        qpar, qperp = Effort.q_par_perp(0.5, cosmo_test, cosmo_ref)

        # These values come from the original test
        # They represent expected AP distortion parameters
        @test qpar ≈ 1.1676180546427928 rtol=3e-5
        @test qperp ≈ 1.1273544308379857 rtol=2e-5

        # Check that q parameters are positive
        @test qpar > 0
        @test qperp > 0
    end

    @testset "q_par_perp: Identity Case" begin
        # When test and reference cosmologies are identical,
        # q_par and q_perp should both be 1

        cosmo = Effort.w0waCDMCosmology(
            ln10Aₛ=3.044, nₛ=0.9649, h=0.6736,
            ωb=0.02237, ωc=0.12, mν=0.06,
            w0=-1.0, wa=0.0
        )

        qpar, qperp = Effort.q_par_perp(0.5, cosmo, cosmo)

        @test qpar ≈ 1.0 atol=1e-10
        @test qperp ≈ 1.0 atol=1e-10
    end

    @testset "q_par_perp: Different Redshifts" begin
        cosmo_test = TEST_COSMO
        cosmo_ref = TEST_COSMO_REF

        # Test at different redshifts (excluding z=0 where AP parameters are not physical)
        for z in [0.5, 1.0, 2.0]
            qpar, qperp = Effort.q_par_perp(z, cosmo_test, cosmo_ref)

            # Basic sanity checks
            @test qpar > 0
            @test qperp > 0
            @test isfinite(qpar)
            @test isfinite(qperp)

            # For the same cosmology models at different z,
            # q parameters should vary (unless cosmologies are identical)
            # But we can't make specific assertions without computing expected values
        end
    end

    @testset "q_par_perp: Physically Reasonable" begin
        cosmo_test = TEST_COSMO
        cosmo_ref = TEST_COSMO_REF

        qpar, qperp = Effort.q_par_perp(0.8, cosmo_test, cosmo_ref)

        # AP distortions are typically modest (within factor of 2)
        # for realistic cosmologies
        @test 0.5 < qpar < 2.0
        @test 0.5 < qperp < 2.0

        # They should be different for different cosmologies
        @test qpar != 1.0 || qperp != 1.0
    end

    @testset "Cosmology Parameter Access" begin
        cosmo = Effort.w0waCDMCosmology(
            ln10Aₛ=3.044, nₛ=0.9649, h=0.6736,
            ωb=0.02237, ωc=0.12, mν=0.06,
            w0=-1.0, wa=0.0
        )

        # Test that all parameters can be accessed
        @test hasfield(typeof(cosmo), :h)
        @test hasfield(typeof(cosmo), :ωb)
        @test hasfield(typeof(cosmo), :ωc)
        @test hasfield(typeof(cosmo), :mν)
        @test hasfield(typeof(cosmo), :w0)
        @test hasfield(typeof(cosmo), :wa)

        # Test parameter values
        @test cosmo.h > 0
        @test cosmo.ωb > 0
        @test cosmo.ωc > 0
        @test cosmo.mν >= 0  # Neutrino mass can be zero
        @test cosmo.w0 < 0   # Dark energy equation of state
    end

    @testset "Cosmology: Edge Cases" begin
        # Test with zero neutrino mass
        cosmo_no_nu = Effort.w0waCDMCosmology(
            ln10Aₛ=3.0, nₛ=0.96, h=0.7,
            ωb=0.022, ωc=0.12, mν=0.0,
            w0=-1.0, wa=0.0
        )
        @test cosmo_no_nu.mν == 0.0

        # Can still compute q parameters
        qpar, qperp = Effort.q_par_perp(0.5, cosmo_no_nu, TEST_COSMO_REF)
        @test isfinite(qpar)
        @test isfinite(qperp)
    end
end
