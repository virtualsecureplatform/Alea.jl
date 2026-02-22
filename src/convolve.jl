using FFTW

export convolve, convolve_fft, convolve_exact,
       convolve_pair, convolve_n,
       normalize_pmf!, negate_dist, dist_stats, mix_two

##################################
# Generic radix-2 Cooley-Tukey FFT
##################################

"""
    generic_fft!(x::Vector{Complex{T}})

In-place radix-2 decimation-in-time FFT for `Vector{Complex{T}}`.
Length must be a power of 2.
"""
function generic_fft!(x::Vector{Complex{T}}) where T<:AbstractFloat
    n = length(x)
    @assert n > 0 && (n & (n - 1)) == 0 "Length must be a power of 2, got $n"

    # Bit-reversal permutation
    j = 0
    for i in 0:(n-2)
        if i < j
            x[i+1], x[j+1] = x[j+1], x[i+1]
        end
        m = n >> 1
        while m >= 1 && j >= m
            j -= m
            m >>= 1
        end
        j += m
    end

    # Butterfly stages
    len = 2
    while len <= n
        half = len >> 1
        angle = T(-2) * T(π) / T(len)
        wn = Complex{T}(cos(angle), sin(angle))
        k = 0
        while k < n
            w = one(Complex{T})
            for m in 0:(half-1)
                u = x[k + m + 1]
                t = w * x[k + m + half + 1]
                x[k + m + 1] = u + t
                x[k + m + half + 1] = u - t
                w *= wn
            end
            k += len
        end
        len <<= 1
    end
    return x
end

"""
    generic_ifft!(x::Vector{Complex{T}})

In-place inverse FFT: conjugate → FFT → conjugate → scale by 1/N.
"""
function generic_ifft!(x::Vector{Complex{T}}) where T<:AbstractFloat
    x .= conj.(x)
    generic_fft!(x)
    x .= conj.(x)
    n = T(length(x))
    x ./= n
    return x
end

##################################
# normalize_pmf!
##################################

"""
    normalize_pmf!(arr::AbstractVector{T}; neg_thresh=-1000*eps(T))

Clamp small negatives (above `neg_thresh`) to zero and renormalize to sum 1.
Large negatives below `neg_thresh` are preserved (indicates a real error).
"""
function normalize_pmf!(arr::AbstractVector{T}; neg_thresh::T = T(-1000) * eps(T)) where T<:AbstractFloat
    @inbounds for i in eachindex(arr)
        if arr[i] < zero(T) && arr[i] > neg_thresh
            arr[i] = zero(T)
        end
    end
    s = sum(arr)
    if s != zero(T)
        arr ./= s
    end
    return arr
end

##################################
# convolve_fft
##################################

"""
    convolve_fft(a::Vector{Float64}, b::Vector{Float64})

FFT-based convolution using FFTW (Float64 fast path).
"""
function convolve_fft(a::Vector{Float64}, b::Vector{Float64})
    n = length(a) + length(b) - 1
    nfft = nextpow(2, n)
    apad = zeros(Float64, nfft)
    bpad = zeros(Float64, nfft)
    apad[1:length(a)] = a
    bpad[1:length(b)] = b
    fa = rfft(apad)
    fb = rfft(bpad)
    c = irfft(fa .* fb, nfft)
    c = c[1:n]
    normalize_pmf!(c)
    return c
end

"""
    convolve_fft(a::Vector{T}, b::Vector{T}) where T<:AbstractFloat

FFT-based convolution using the generic radix-2 FFT kernel.
Works with any AbstractFloat type (Double64, BigFloat, etc.).
"""
function convolve_fft(a::Vector{T}, b::Vector{T}) where T<:AbstractFloat
    n = length(a) + length(b) - 1
    nfft = nextpow(2, n)
    apad = zeros(Complex{T}, nfft)
    bpad = zeros(Complex{T}, nfft)
    @inbounds for i in eachindex(a)
        apad[i] = Complex{T}(a[i], zero(T))
    end
    @inbounds for i in eachindex(b)
        bpad[i] = Complex{T}(b[i], zero(T))
    end
    generic_fft!(apad)
    generic_fft!(bpad)
    apad .*= bpad
    generic_ifft!(apad)
    c = Vector{T}(undef, n)
    @inbounds for i in 1:n
        c[i] = real(apad[i])
    end
    normalize_pmf!(c)
    return c
end

##################################
# convolve_exact
##################################

"""
    convolve_exact(a::Vector{T}, b::Vector{T}) where T<:AbstractFloat

Direct convolution (O(n*m)). Exact up to floating-point arithmetic.
"""
function convolve_exact(a::Vector{T}, b::Vector{T}) where T<:AbstractFloat
    n = length(a) + length(b) - 1
    c = zeros(T, n)
    @inbounds for i in eachindex(a)
        ai = a[i]
        if ai != zero(T)
            for j in eachindex(b)
                c[i + j - 1] += ai * b[j]
            end
        end
    end
    normalize_pmf!(c)
    return c
end

##################################
# Dispatcher and composition
##################################

"""
    convolve(a::Vector{T}, b::Vector{T}) where T<:AbstractFloat

Dispatch to `convolve_fft` or `convolve_exact` based on `conv_mode()`.
"""
function convolve(a::Vector{T}, b::Vector{T}) where T<:AbstractFloat
    if conv_mode() == :fft
        return convolve_fft(a, b)
    else
        return convolve_exact(a, b)
    end
end

"""
    convolve_pair(a::Vector{T}, off_a::Int, b::Vector{T}, off_b::Int)

Convolve two offset-PMFs, returning `(result_array, off_a + off_b)`.
"""
function convolve_pair(a::Vector{T}, off_a::Int, b::Vector{T}, off_b::Int) where T<:AbstractFloat
    return convolve(a, b), off_a + off_b
end

"""
    convolve_n(arr::Vector{T}, off::Int, n::Int)

Convolve `arr` with itself `n` times using binary exponentiation.
Returns `(result_array, off * n)`.
"""
function convolve_n(arr::Vector{T}, off::Int, n::Int) where T<:AbstractFloat
    if n == 0
        return T[one(T)], 0
    end
    res = T[one(T)]
    base = arr
    m = n
    while m > 0
        if (m & 1) == 1
            res = convolve(res, base)
        end
        m >>= 1
        if m > 0
            base = convolve(base, base)
        end
    end
    return res, off * n
end

##################################
# Utility functions
##################################

"""
    negate_dist(arr::Vector{T}, off::Int)

Negate a distribution: reverse the PMF array and negate the offset range.
If X ~ arr with offset `off`, then -X ~ reverse(arr) with offset `-(off + len - 1)`.
"""
function negate_dist(arr::Vector{T}, off::Int) where T<:AbstractFloat
    new_off = -(off + length(arr) - 1)
    return reverse(arr), new_off
end

"""
    dist_stats(arr::Vector{T}, off::Int)

Compute mean, variance, std, total probability, tail probabilities (1σ–5σ),
and support bounds for an offset-PMF.
"""
function dist_stats(arr::Vector{T}, off::Int) where T<:AbstractFloat
    total = sum(arr)
    if total == zero(T)
        return (mean=zero(T), variance=zero(T), std=zero(T), total=zero(T),
                tails=Dict{Int, T}(), min=0, max=0)
    end
    mn = zero(T)
    for i in eachindex(arr)
        mn += T(off + (i - 1)) * arr[i]
    end
    mn /= total
    vr = zero(T)
    for i in eachindex(arr)
        v = T(off + (i - 1)) - mn
        vr += v * v * arr[i]
    end
    vr /= total
    sd = sqrt(vr)
    tails = Dict{Int, T}()
    for mult in 1:5
        thresh = T(mult) * sd
        tail = zero(T)
        for i in eachindex(arr)
            v = T(off + (i - 1))
            if abs(v - mn) > thresh
                tail += arr[i]
            end
        end
        tails[mult] = tail / total
    end
    minv = off
    maxv = off + length(arr) - 1
    return (mean=mn, variance=vr, std=sd, total=total, tails=tails, min=minv, max=maxv)
end

"""
    mix_two(a::Vector{T}, off_a::Int, b::Vector{T}, off_b::Int, wa::T)

Weighted mixture of two offset-PMFs: `wa * a + (1 - wa) * b`.
"""
function mix_two(a::Vector{T}, off_a::Int, b::Vector{T}, off_b::Int, wa::T) where T<:AbstractFloat
    wb = one(T) - wa
    minv = min(off_a, off_b)
    maxv = max(off_a + length(a) - 1, off_b + length(b) - 1)
    out = zeros(T, maxv - minv + 1)
    for i in eachindex(a)
        out[(off_a - minv) + i] += wa * a[i]
    end
    for i in eachindex(b)
        out[(off_b - minv) + i] += wb * b[i]
    end
    normalize_pmf!(out)
    return out, minv
end
