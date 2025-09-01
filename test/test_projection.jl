using Test
using Effort
using Zygote
using ForwardDiff
include("test_helpers.jl")

@testset "Projection functions" begin
    k_vals = [0.1, 0.5, 1.0]
    μ_vals = [-1.0, 0.0, 0.5, 1.0]
    mono_vals = [100.0, 50.0, 25.0]
    quad_vals = [20.0, 10.0, 5.0]
    hexa_vals = [4.0, 2.0, 1.0]
    q_par = 1.2
    q_perp = 0.8
    F = q_par / q_perp
    
    Int_Mono, Int_Quad, Int_Hexa = Effort.interp_Pℓs(mono_vals, quad_vals, hexa_vals, k_vals)
    
    @test Effort._Pkμ(0.5, 0.0, Int_Mono, Int_Quad, Int_Hexa) ≈ Int_Mono(0.5) * Effort._Legendre_0(0.0) + Int_Quad(0.5) * Effort._Legendre_2(0.0) + Int_Hexa(0.5) * Effort._Legendre_4(0.0)
    
    @test Effort._k_true(1.0, 0.0, q_perp, F) ≈ 1.0 / q_perp
    @test Effort._k_true(1.0, 1.0, q_perp, F) ≈ 1.0 / q_perp * sqrt(1 + (1 / F^2 - 1))
    
    @test Effort._μ_true(0.0, F) ≈ 0.0
    @test Effort._μ_true(1.0, F) ≈ 1.0 / F / sqrt(1 + (1 / F^2 - 1))
    
    k_array = [0.5, 1.0]
    μ_array = [0.0, 0.5]
    k_true_array = Effort._k_true(k_array, μ_array, q_perp, F)
    @test length(k_true_array) == length(k_array) * length(μ_array)
    
    μ_true_array = Effort._μ_true(μ_array, F)
    @test length(μ_true_array) == length(μ_array)
    
    P_obs_val = Effort._P_obs(0.5, 0.5, q_par, q_perp, Int_Mono, Int_Quad, Int_Hexa)
    @test P_obs_val > 0.0
    
    qpar_test, qperp_test = Effort.q_par_perp(1.0, mycosmo, mycosmo)
    @test isapprox(qpar_test, 1.0, rtol=1e-10)
    @test isapprox(qperp_test, 1.0, rtol=1e-10)
    
    mono_matrix = reshape(mono_vals, :, 1)
    quad_matrix = reshape(quad_vals, :, 1)
    hexa_matrix = reshape(hexa_vals, :, 1)
    l0_vec = Effort._Legendre_0.(μ_vals)
    l2_vec = Effort._Legendre_2.(μ_vals)
    l4_vec = Effort._Legendre_4.(μ_vals)
    
    Pk_recon = Effort._Pk_recon(mono_matrix, quad_matrix, hexa_matrix, l0_vec, l2_vec, l4_vec)
    @test size(Pk_recon) == (length(k_vals), length(μ_vals))
    @test all(Pk_recon .> 0.0)
    
    W_4d = rand(2, 3, 2, 3)
    v_matrix = rand(3, 3)
    result_4d = Effort.window_convolution(W_4d, v_matrix)
    @test size(result_4d) == (2, 2)
    
    W_matrix = rand(3, 4)
    v_vector = rand(4)
    result_matrix = Effort.window_convolution(W_matrix, v_vector)
    @test size(result_matrix) == (3,)
    @test result_matrix ≈ W_matrix * v_vector
end

@testset "AP application tests" begin
    k_input = [0.1, 0.5, 1.0, 2.0]
    k_output = [0.2, 0.8]
    mono_in = [200.0, 100.0, 50.0, 25.0]
    quad_in = [40.0, 20.0, 10.0, 5.0]
    hexa_in = [8.0, 4.0, 2.0, 1.0]
    q_par = 1.1
    q_perp = 0.9
    
    P0_obs, P2_obs, P4_obs = Effort.apply_AP(k_input, k_output, mono_in, quad_in, hexa_in, q_par, q_perp; n_GL_points=5)
    @test length(P0_obs) == length(k_output)
    @test length(P2_obs) == length(k_output)
    @test length(P4_obs) == length(k_output)
    @test all(P0_obs .> 0.0)
    
    P0_check, P2_check, P4_check = Effort.apply_AP_check(k_input, k_output, mono_in, quad_in, hexa_in, q_par, q_perp)
    @test length(P0_check) == length(k_output)
    @test length(P2_check) == length(k_output)
    @test length(P4_check) == length(k_output)
    @test all(P0_check .> 0.0)
end

@testset "AP convergence tests" begin
    a_18, b_18, c_18 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=18)
    a_72, b_72, c_72 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=72)
    a_126, b_126, c_126 = Effort.apply_AP(myx, myx, monotest, quadtest, hexatest, q_par, q_perp; n_GL_points=126)
    a_c, b_c, c_c = Effort.apply_AP_check(myx, myx, monotest, quadtest, hexatest, q_par, q_perp)
    
    @test isapprox(a_18, a_c, rtol=1e-5)
    @test isapprox(b_18, b_c, rtol=1e-5)
    @test isapprox(c_18, c_c, rtol=1e-5)
    
    @test isapprox(a_72, a_c, rtol=1e-6)
    @test isapprox(b_72, b_c, rtol=1e-6)
    @test isapprox(c_72, c_c, rtol=1e-6)
    
    @test isapprox(a_126, a_c, rtol=1e-7)
    @test isapprox(b_126, b_c, rtol=1e-7)
    @test isapprox(c_126, c_c, rtol=1e-7)
end

@testset "AP with real data" begin
    mycosmo_ref = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
    qpar, qperp = Effort.q_par_perp(0.5, mycosmo, mycosmo_ref)
    @test isapprox(qpar, 1.1676180546427928, rtol=3e-5)
    @test isapprox(qperp, 1.1273544308379857, rtol=2e-5)
    a, b, c = Effort.apply_AP(k, k_test_data, Pℓ[1, :], Pℓ[2, :], Pℓ[3, :], qpar, qperp; n_GL_points=5)
    @test isapprox(a[4:end], Pℓ_AP[1, 4:end], rtol=5e-4)
    @test isapprox(b[4:end], Pℓ_AP[2, 4:end], rtol=5e-4)
    @test isapprox(c[4:end], Pℓ_AP[3, 4:end], rtol=5e-4)
end

@testset "apply_AP AD compatibility" begin
    # Use real data from Zenodo
    mono_real = Pℓ[1, :]  # Monopole from Zenodo data
    quad_real = Pℓ[2, :]  # Quadrupole from Zenodo data
    hexa_real = Pℓ[3, :]  # Hexadecapole from Zenodo data
    k_out_subset = k_test_data[1:5]  # Use subset of output k for faster tests
    
    @testset "Gradient w.r.t multipoles" begin
        # Test monopole gradient
        loss_mono(m) = sum(Effort.apply_AP(k, k_out_subset, m, quad_real, hexa_real, 1.05, 0.95; n_GL_points=5)[1])
        grad_z = Zygote.gradient(loss_mono, mono_real)[1]
        grad_fd = ForwardDiff.gradient(loss_mono, mono_real)
        @test isapprox(grad_z, grad_fd, rtol=1e-8, atol=1e-10)
        
        # Test quadrupole gradient
        loss_quad(q) = sum(Effort.apply_AP(k, k_out_subset, mono_real, q, hexa_real, 1.05, 0.95; n_GL_points=5)[2])
        grad_z = Zygote.gradient(loss_quad, quad_real)[1]
        grad_fd = ForwardDiff.gradient(loss_quad, quad_real)
        @test isapprox(grad_z, grad_fd, rtol=1e-8, atol=1e-10)
        
        # Test hexadecapole gradient
        loss_hexa(h) = sum(Effort.apply_AP(k, k_out_subset, mono_real, quad_real, h, 1.05, 0.95; n_GL_points=5)[3])
        grad_z = Zygote.gradient(loss_hexa, hexa_real)[1]
        grad_fd = ForwardDiff.gradient(loss_hexa, hexa_real)
        @test isapprox(grad_z, grad_fd, rtol=1e-8, atol=1e-10)
    end
    
    @testset "Gradient w.r.t AP parameters" begin
        function loss_ap(params)
            P0, P2, P4 = Effort.apply_AP(k, k_out_subset, mono_real, quad_real, hexa_real, 
                                         params[1], params[2]; n_GL_points=5)
            return sum(P0) + sum(P2) + sum(P4)
        end
        
        params = [1.1, 0.9]
        grad_z = Zygote.gradient(loss_ap, params)[1]
        grad_fd = ForwardDiff.gradient(loss_ap, params)
        @test isapprox(grad_z, grad_fd, rtol=1e-8, atol=1e-10)
        @test all(isfinite.(grad_z))
    end
    
    @testset "Combined gradient with Zenodo data" begin
        function loss_all(x)
            n = length(k)
            m, q, h = x[1:n], x[n+1:2n], x[2n+1:3n]
            qpar, qperp = x[3n+1], x[3n+2]
            P0, P2, P4 = Effort.apply_AP(k, k_out_subset, m, q, h, qpar, qperp; n_GL_points=5)
            return sum(P0) + 0.5*sum(P2) + 0.25*sum(P4)
        end
        
        x_all = vcat(mono_real, quad_real, hexa_real, [1.05, 0.95])
        grad_z = Zygote.gradient(loss_all, x_all)[1]
        grad_fd = ForwardDiff.gradient(loss_all, x_all)
        @test isapprox(grad_z, grad_fd, rtol=1e-8, atol=1e-10)
    end
    
    @testset "Gradient with actual AP parameters from test" begin
        # Use the actual AP parameters computed in the "AP with real data" test
        mycosmo_ref = Effort.w0waCDMCosmology(ln10Aₛ=3.0, nₛ=0.96, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0)
        qpar, qperp = Effort.q_par_perp(0.5, mycosmo, mycosmo_ref)
        
        function loss_real_ap(multipoles)
            n = length(k)
            P0, P2, P4 = Effort.apply_AP(k, k_test_data[1:5], multipoles[1:n], multipoles[n+1:2n], 
                                         multipoles[2n+1:3n], qpar, qperp; n_GL_points=5)
            return sum(P0) + sum(P2) + sum(P4)
        end
        
        x_multipoles = vcat(Pℓ[1, :], Pℓ[2, :], Pℓ[3, :])
        grad_z = Zygote.gradient(loss_real_ap, x_multipoles)[1]
        grad_fd = ForwardDiff.gradient(loss_real_ap, x_multipoles)
        @test isapprox(grad_z, grad_fd, rtol=1e-8, atol=1e-10)
    end
end