using QuadGK, Trapz

rho_0 = 37957104814.59015;

# =========================================================================================

function E(x, z)

    return sqrt(x["om0"] * (1 + z)^3 + (1 - x["om0"]))

end


function D_plus(x, z)

    integrand(zz) = (1 + zz)/(x["om0"] * (1 + zz)^3 + (1 - x["om0"]))^(3/2)

    I = quadgk(integrand, z, 1e5, rtol = 1e-8)[1]

    return 5.0/2 * x["om0"] * E(x, z) * I

end


function d(x, z)

    return D_plus(x, z)/D_plus(x, 0.0)

end


function T_we(k, x)

    s = 44.5 * log(9.83/(x["om0"]*x["h"]^2)) / sqrt(1.0 + 10.0 * (x["om0"] * x["h"] * x["h"])^0.75)

    alphaGamma = 1.0 - 0.328 * log(431.0 * x["om0"] * x["h"] * x["h"]) * x["omb"] / x["om0"] +
                   0.38 * log(22.3 * x["om0"] * x["h"] * x["h"]) * (x["omb"] / x["om0"]) * (x["omb"] / x["om0"])

    Gamma = x["om0"] * x["h"] * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k * x["h"] * s)^4.0))

    q = k * (x["Tcmb0"]/2.7)^2 / Gamma

    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)

    L0 = log(2.0 * exp(1.0) + 1.8 * q)

    return L0 / (L0 + C0 * q * q)

end


function M2R(M, x, z)

    d = x["om0"] * (1 + z)^3 / (x["om0"] * (1 + z)^3 + (1 - x["om0"])) - 1

    DELTA_V = 200/(1 + d)

    return (3.0 * M / (4.0 * pi * rho_0))^(0.333333)

end


function R2M(R, x, z)

    d = x["om0"] * (1 + z)^3 / (x["om0"] * (1 + z)^3 + (1 - x["om0"])) - 1

    DELTA_V = 200/(1 + d)

    return (4/3 * pi * rho_0) * R^3

end


function W(M, k, z, x)

    return 3.0 * (sin(k * M2R(M, x, z)) - (k*M2R(M, x, z)) * cos(k*M2R(M, x, z)))/(k*M2R(M, x, z))^3.0

end


function A_norm(x, z)

    M8 = R2M(8, x, z)

    k_ = range(1e-4, 3.0, length = 200)

    W_ = [W(M8, kk, z, x) for kk in k_]

    T_ = [T_we(kk, x) for kk in k_]

    d = 1/(2 * pi^2) * trapz(k_, k_.^(x["ns"]+2) .* T_.^2 .* W_.^2)

    return x["s8"]^2 / d

end


function PS(om0, k, z)

    xx = Dict("h" => 0.6714, "om0" => om0, "omb" => 0.049,
         "ns" => 0.9624, "s8" => 0.83, "Tcmb0" => 2.75)

    T_ = T_we(k, xx)
    norm = A_norm(xx, z)

    return k^xx["ns"] * T_^2 * norm * d(xx, z)^2

end


function PS_vec(x0, x1, x2, x3, x4, k, z)

    xx = Dict("h" => x3, "om0" => x0, "omb" => x1,
              "ns" => x2, "s8" => x4, "Tcmb0" => 2.75)

    T_ = T_we(k, xx)
    norm = A_norm(xx, z)

    return k^xx["ns"] * T_^2 * norm * d(xx, z)^2

end
