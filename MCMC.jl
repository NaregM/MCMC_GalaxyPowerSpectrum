using Random
using Distributions, HypothesisTests
using Statistics, Bootstrap, LinearAlgebra
include("cosmology.jl")

# ====================================================================================

function SPDmatrix(p, n)

    A = rand(n, n)
    A = 0.5 * (A + A')

    return p * (A + n * I(n))

end

# Parameter range
struct P

    p_low
    p_high

end

# log-Priors
prior(x, P) = log(pdf(Uniform(P.p_low, P.p_high), x))

# log-Likelihood
likelihood(x0, x1, x2, x3, x4, data, data_err, data_k) = +([log(exp(-0.5 * (data[i] - 6.6 * 6.6 * PS_vec(x0, x1, x2, x3, x4, kk, 0.23))^2 / (data_err[i]^2))) - log(data_err[i]) for (i, kk) in enumerate(data_k)]...)

# Posterior
posterior(x0, x1, x2, x3, x4, P0, P1, P2, P3, P4, data, data_err, data_k) = likelihood(x0, x1, x2, x3, x4, data, data_err, data_k) + prior(x0, P0) + prior(x1, P1) + prior(x2, P2) + prior(x3, P3) + prior(x4, P4) # prior0(x0) + prior1(x1)+ prior2(x2) + prior3(x3) + prior4(x4)

# proposal density,  1e-6 is not good,
σxy = SPDmatrix(1e-6, 5)          #1.0e-5 .* [19.0 5.8 8.0 2.2 4.4;

foldedNormalPDF(xx0, xx1, xx2, xx3, xx4, m0, m1 , m2, m3, m4) = pdf(MvNormal([m0, m1, m2, m3, m4], σxy), [xx0, xx1, xx2, xx3, xx4])


foldedNormalRV(m0, m1, m2, m3, m4) = rand(MvNormal([m0, m1, m2, m3, m4], σxy))

# ===================================================================================================
# Sampler
function mySampler4(f, q, qrv, P0, P1, P2, P3, P4, data, data_err, data_k, N_samples)

    x0 = 0.5    #0.32
    x1 = 0.025  #0.025
    x2 = 0.5    #0.98
    x3 = 0.5
    x4 = 0.5

    warmN, N = N_samples, N_samples + 1000

    samples_x0 = zeros(Float64, N - warmN)
    samples_x1 = zeros(Float64, N - warmN)
    samples_x2 = zeros(Float64, N - warmN)
    samples_x3 = zeros(Float64, N - warmN)
    samples_x4 = zeros(Float64, N - warmN)

    @showprogress 1 "Sampling..." for t in 1:N

        while true

            x0_star, x1_star, x2_star, x3_star, x4_star = qrv(x0, x1, x2, x3, x4)

            L = exp(f(x0_star, x1_star, x2_star, x3_star, x4_star, P0, P1, P2, P3, P4, data, data_err, data_k))/exp(f(x0, x1, x2, x3, x4, P0, P1, P2, P3, P4, data, data_err, data_k))

            H = min(1, L * q(x0, x1, x2, x3, x4, x0_star, x1_star, x2_star, x3_star, x4_star)/q(x0_star, x1_star, x2_star, x3_star, x4_star, x0, x1, x2, x3, x4))

            if rand() < H

                x0 = x0_star
                x1 = x1_star
                x2 = x2_star
                x3 = x3_star
                x4 = x4_star

                if t > warmN

                    samples_x0[t - warmN] = x0
                    samples_x1[t - warmN] = x1
                    samples_x2[t - warmN] = x2
                    samples_x3[t - warmN] = x3
                    samples_x4[t - warmN] = x4

                end

                break

            end
        end
    end

    return samples_x0, samples_x1, samples_x2, samples_x3, samples_x4

end
