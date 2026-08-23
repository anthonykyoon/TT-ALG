#!/usr/bin/env julia
# MPS.jl — 1D Hubbard ground state via DMRG (ITensors + ITensorMPS)
# - Uses OpSum (modern replacement of AutoMPO)
# - Explicit loops (no summation symbol)
# - Product-state initialization (works with QNs)
# - Local expectations via expect(psi, "OpName") or expect(...; sites=i:i)

using ITensors
using ITensorMPS

# ---------- CLI parsing (positional): N  t  U  mu  periodic ----------
function parse_args()
    N         = length(ARGS) >= 1 ? parse(Int, ARGS[1])      : 20
    t         = length(ARGS) >= 2 ? parse(Float64, ARGS[2])  : 1.0
    U         = length(ARGS) >= 3 ? parse(Float64, ARGS[3])  : 4.0
    mu        = length(ARGS) >= 4 ? parse(Float64, ARGS[4])  : 0.0
    periodic  = length(ARGS) >= 5 ? lowercase(ARGS[5]) in ("true","t","1","yes","y") : false
    return (N, t, U, mu, periodic)
end

# ---------- Build Hubbard MPO with loops ----------
function hubbard_mpo(sites, t::Float64, U::Float64, mu::Float64; periodic::Bool=false)
    N = length(sites)
    ops = OpSum()

    # nearest-neighbor hopping (open BC by default)
    for i in 1:N-1
        add!(ops, -t, "Cdagup", i, "Cup", i+1)
        add!(ops, -t, "Cdagup", i+1, "Cup", i)
        add!(ops, -t, "Cdagdn", i, "Cdn", i+1)
        add!(ops, -t, "Cdagdn", i+1, "Cdn", i)
    end
    # periodic wrap-around
    if periodic && N > 2
        add!(ops, -t, "Cdagup", N, "Cup", 1)
        add!(ops, -t, "Cdagup", 1, "Cup", N)
        add!(ops, -t, "Cdagdn", N, "Cdn", 1)
        add!(ops, -t, "Cdagdn", 1, "Cdn", N)
    end

    # on-site interaction and chemical potential
    for i in 1:N
        add!(ops, U, "Nupdn", i)
        if mu != 0.0
            add!(ops, -mu, "Nup", i)
            add!(ops, -mu, "Ndn", i)
        end
    end

    return MPO(ops, sites)
end

function main()
    N, t, U, mu, periodic = parse_args()
    println("Hubbard chain: N=$N, t=$t, U=$U, mu=$mu, periodic=$periodic")

    # Electron sites with QNs (tracks total N and Sz)
    sites = siteinds("Electron", N; conserve_qns=true)

    # Hamiltonian
    H = hubbard_mpo(sites, t, U, mu; periodic=periodic)

    # Initial MPS in a definite QN sector (half-filling, alternating spins)
    st = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    psi0 = productMPS(sites, st)

    # DMRG sweeps
    sweeps = Sweeps(6)
    maxdim!(sweeps, 64, 128, 256, 256, 400, 600)
    cutoff!(sweeps, 1e-8)
    noise!(sweeps, 1e-6, 1e-7, 0.0, 0.0, 0.0, 0.0)

    # Run DMRG (use outputlevel=0 to be quiet)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    println()
    println("Ground state energy approx ", energy)

    # --- Local observables ---
    # Preferred path: many versions infer sites from psi and return a Vector
    nup   = expect(psi, "Nup")
    ndn   = expect(psi, "Ndn")
    ndoub = expect(psi, "Nupdn")

    # If your local API returns empty vectors, fall back to per-site calls:
    if length(nup) != N || length(ndn) != N || length(ndoub) != N
        nup   = [expect(psi, "Nup";   sites=i:i)[1]   for i in 1:N]
        ndn   = [expect(psi, "Ndn";   sites=i:i)[1]   for i in 1:N]
        ndoub = [expect(psi, "Nupdn"; sites=i:i)[1]   for i in 1:N]
    end

    Sz = 0.5 .* (nup .- ndn)

    println()
    println("<n_up> per site:   ", nup)
    println("<n_dn> per site:   ", ndn)
    println("<double> per site: ", ndoub)
    println("<Sz> per site:     ", Sz)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
