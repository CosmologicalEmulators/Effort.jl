using Test
using Effort
using ForwardDiff
using Random

@testset "Built-in Bias Combination Jacobian Pairs" begin
    @test keys(Effort.BUILTIN_BIAS_COMBINATIONS) == keys(Effort.BUILTIN_JAC_BIAS_COMBINATIONS)

    rng = Random.MersenneTwister(314159)
    bias_vectors = [
        randn(rng, 11),
        2 .* rand(rng, 11) .- 1,
        [1.7, -0.4, 0.2, 0.8, -0.1, 0.3, -0.2, 0.75, 0.1, -0.05, 0.02],
    ]

    for name in sort(collect(keys(Effort.BUILTIN_BIAS_COMBINATIONS)))
        @testset "$name" begin
            bias_combination = Effort.BUILTIN_BIAS_COMBINATIONS[name]
            jacobian_bias_combination = Effort.BUILTIN_JAC_BIAS_COMBINATIONS[name]

            for bs in bias_vectors
                J_auto = ForwardDiff.jacobian(bias_combination, bs)
                J_manual = jacobian_bias_combination(bs)

                @test size(J_manual) == size(J_auto)
                @test J_manual ≈ J_auto rtol=1e-10 atol=1e-10
            end
        end
    end
end

@testset "Bias Combination Jacobian Correctness" begin
    # Iterate over all trained emulators available in the global dictionary
    for (emu_name, emu_dict) in Effort.trained_emulators
        @testset "Emulator: $emu_name" begin
            for (multipole, emu) in emu_dict
                @testset "Multipole ℓ=$multipole" begin
                    # Generate random bias parameters
                    # We use a fixed seed for reproducibility, but the test should hold generically
                    rng = Random.MersenneTwister(42)
                    bs = rand(rng, 11)

                    # 1. Compute Jacobian using Automatic Differentiation (Ground Truth)
                    # We wrap the BiasCombination function to be compatible with ForwardDiff
                    f_biases = x -> emu.BiasCombination(x)
                    J_auto = ForwardDiff.jacobian(f_biases, bs)

                    # 2. Compute Jacobian using the Emulator's built-in analytical/sparse method
                    J_manual = emu.JacobianBiasCombination(bs)

                    # 3. Compare
                    # Check dimensions
                    @test size(J_manual) == size(J_auto)

                    # Check values
                    # Using a tolerance. Since these are likely linear combinations,
                    # precision should be quite high (close to machine epsilon),
                    # but we'll use a safe tolerance.
                    @test J_manual ≈ J_auto rtol=1e-10 atol=1e-10
                end
            end
        end
    end
end
