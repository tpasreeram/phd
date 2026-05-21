# vim: ts=4 sts=4 sw=4 et
using QuantumToolbox
using LinearAlgebra

function local_ops(N)
    sp_list = Vector{QuantumObject}()
    for i in 1:N
        ops = fill(qeye(2), N)
        ops[i] = sigmap()
        push!(sp_list, tensor(ops...))
    end
    return sp_list
end

function build_ops(gmat, N, tol=1e-10)
    Gamma, alpha = eigen(gmat)
    ops_list = local_ops(N)

    c_ops = []
    for nu in 1:N
        if Gamma[nu] > tol
            O_nu = zero(ops_list(1))
            for i in 1:N
                O_nu += alpha[i, nu] * ops_list[i]
            end
            push!(c_ops, sqrt(Gamma[nu]) * O_nu)
        end
    end
    return c_ops
end

function perp(N, k, a)
    gmat = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        if i == j
            gmat[i, j] = 1
        else
            r = abs(i - j) * a
            x = k * r
            gmat[i, j] = (3/2) * (sin(x)/x + cos(x)/x^2 - sin(x)/x^3)
        end
    end
    return gmat
end

function parallel(N, k, a)
    gmat = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        if i == j
            gmat[i, j] = 1
        else
            r = abs(i - j) * a
            x = k * r
            gmat[i, j] = 3 * (sin(x)/x^3 - cos(x)/x^2)
        end
    end
    return gmat
end

function main()
    N = 6
    tlist = range(0, 2, length=500)

    H = 0 * tensor(fill(qeye(2), N)...)
    rho0 = tensor(fill(basis(2, 1), N)...)

    a = parallel(N, 1, 0.1)
    b = perp(N, 1, 0.1)

    gmat_list = [a, b]

    for g in gmat_list
        c_ops = build_ops(g, N)

        result = mesolve(H, rho0, tlist, c_ops, e_ops = [dagger(c) * c for c in c_ops])

        I_t = sum(result.expect, dims=1)
        I_t ./= I_t[1]

        plot!(tlist, vec(I_t), label="")
    end

    xlabel!("time")
    ylabel!("normalized photon emission rate")
    display(current())
end

main()
