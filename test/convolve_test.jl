using Test
using Alea

@testset "convolve" begin

    @testset "normalize_pmf!" begin
        # Basic normalization
        arr = [0.3, 0.3, 0.4]
        normalize_pmf!(arr)
        @test sum(arr) ≈ 1.0

        # Clamp small negatives to zero
        arr = [0.5, -1e-15, 0.5]
        normalize_pmf!(arr)
        @test arr[2] == 0.0
        @test sum(arr) ≈ 1.0

        # Large negatives preserved (indicates a real problem)
        arr = [0.5, -0.1, 0.6]
        normalize_pmf!(arr)
        @test arr[2] < 0.0

        # All-zero array stays zero
        arr = zeros(5)
        normalize_pmf!(arr)
        @test all(arr .== 0.0)
    end

    @testset "convolve_exact" begin
        # Uniform * Uniform = Triangular
        u = [0.5, 0.5]
        result = convolve_exact(u, u)
        @test length(result) == 3
        @test result ≈ [0.25, 0.5, 0.25]

        # Delta * anything = identity
        delta = [1.0]
        a = [0.2, 0.3, 0.5]
        @test convolve_exact(delta, a) ≈ a

        # Commutative
        b = [0.4, 0.6]
        @test convolve_exact(a, b) ≈ convolve_exact(b, a)
    end

    @testset "convolve_fft Float64" begin
        # Matches exact within tolerance
        a = [0.2, 0.3, 0.5]
        b = [0.4, 0.6]
        exact = convolve_exact(a, b)
        fft_result = convolve_fft(a, b)
        @test length(fft_result) == length(exact)
        @test fft_result ≈ exact atol=1e-12

        # Uniform * Uniform
        u = [0.5, 0.5]
        result = convolve_fft(u, u)
        @test result ≈ [0.25, 0.5, 0.25] atol=1e-14
    end

    @testset "convolve dispatcher" begin
        a = [0.5, 0.5]
        b = [0.5, 0.5]

        Alea.set_conv_mode!(:exact)
        r_exact = convolve(a, b)
        Alea.set_conv_mode!(:fft)
        r_fft = convolve(a, b)
        @test r_exact ≈ r_fft atol=1e-12
    end

    @testset "convolve_pair" begin
        a = [0.5, 0.5]
        b = [0.3, 0.7]
        arr, off = convolve_pair(a, 10, b, -3)
        @test off == 7
        @test length(arr) == 3
        @test sum(arr) ≈ 1.0
    end

    @testset "convolve_n" begin
        # n=0 → delta
        arr, off = convolve_n([0.5, 0.5], 1, 0)
        @test arr == [1.0]
        @test off == 0

        # n=1 → identity
        a = [0.3, 0.7]
        arr, off = convolve_n(a, 5, 1)
        @test arr ≈ a
        @test off == 5

        # 4-fold Bernoulli = Binomial(4, 0.5)
        Alea.set_conv_mode!(:fft)
        coin = [0.5, 0.5]
        arr, off = convolve_n(coin, 0, 4)
        @test off == 0
        @test length(arr) == 5
        expected = [1/16, 4/16, 6/16, 4/16, 1/16]
        @test arr ≈ expected atol=1e-12

        # Same result with exact mode
        Alea.set_conv_mode!(:exact)
        arr_e, off_e = convolve_n(coin, 0, 4)
        @test off_e == 0
        @test arr_e ≈ expected
        Alea.set_conv_mode!(:fft)  # restore
    end

    @testset "negate_dist" begin
        a = [0.2, 0.3, 0.5]
        arr, off = negate_dist(a, 10)
        @test arr == reverse(a)
        @test off == -(10 + 2)  # -(off + len - 1) = -12

        # Negate of negate = original
        arr2, off2 = negate_dist(arr, off)
        @test arr2 ≈ a
        @test off2 == 10
    end

    @testset "dist_stats" begin
        # Symmetric around 0: Rademacher {-1, +1} with p=0.5
        arr = [0.5, 0.0, 0.5]
        s = dist_stats(arr, -1)
        @test s.mean ≈ 0.0 atol=1e-15
        @test s.variance ≈ 1.0
        @test s.std ≈ 1.0
        @test s.total ≈ 1.0
        @test s.min == -1
        @test s.max == 1

        # All mass at one point
        arr = [1.0]
        s = dist_stats(arr, 42)
        @test s.mean ≈ 42.0
        @test s.variance ≈ 0.0
        @test s.std ≈ 0.0

        # Zero array
        arr = zeros(5)
        s = dist_stats(arr, 0)
        @test s.mean == 0.0
        @test s.total == 0.0
    end

    @testset "mix_two" begin
        a = [1.0]  # delta at 0
        b = [1.0]  # delta at 10
        arr, off = mix_two(a, 0, b, 10, 0.3)
        @test off == 0
        @test length(arr) == 11
        @test arr[1] ≈ 0.3 atol=1e-14
        @test arr[11] ≈ 0.7 atol=1e-14
        @test sum(arr) ≈ 1.0

        # Equal weight mixing of identical dists
        c = [0.25, 0.5, 0.25]
        arr2, off2 = mix_two(c, 0, c, 0, 0.5)
        @test arr2 ≈ c
        @test off2 == 0
    end

    @testset "generic FFT (non-Float64)" begin
        # Test with Float32 to verify generic path works
        a = Float32[0.5, 0.5]
        b = Float32[0.5, 0.5]
        result = convolve_fft(a, b)
        @test eltype(result) == Float32
        @test length(result) == 3
        @test Float64.(result) ≈ [0.25, 0.5, 0.25] atol=1e-6

        # convolve_n with Float32
        arr, off = convolve_n(a, 0, 4)
        @test eltype(arr) == Float32
        expected = Float32[1/16, 4/16, 6/16, 4/16, 1/16]
        @test arr ≈ expected atol=1e-5
    end

    @testset "Double64 FFT (if available)" begin
        try
            # Try to load DoubleFloats — skip gracefully if unavailable
            @eval using DoubleFloats
            a = @eval Double64[0.5, 0.5]
            r = convolve_fft(a, a)
            @test eltype(r) == @eval Double64
            @test Float64.(r) ≈ [0.25, 0.5, 0.25] atol=1e-28

            # convolve_n with Double64
            arr, off = convolve_n(a, 0, 4)
            @test eltype(arr) == @eval Double64
            expected = [1/16, 4/16, 6/16, 4/16, 1/16]
            @test Float64.(arr) ≈ expected atol=1e-28
        catch e
            @info "DoubleFloats not available, skipping Double64 tests" exception=e
        end
    end

end
