function householder(x::Vector{Float64})
    x_norm = norm(x)
    n = length(x)

    if x_norm == 0.0
        return Matrix(I, n, n)
    end

    v = copy(x)
    v[1] += sign(x[1]) * x_norm
    v = v / norm(v)  # Normalize so that v^T v = 1
    H = I - 2.0 * (v * v')  # Householder reflector
    return H
end

function golub_kahan_bidiagonalization(A::Matrix{Float64})
    m, n = size(A)
    U = Matrix(I, m, m)
    V = Matrix(I, n, n)
    B = copy(A)

    for k in 1:n
        # Left Householder
        H = Matrix(I, m, m)
        H_k = householder(B[k:end, k])
        H[k:end, k:end] = H_k
        B = H * B
        U = U * H

        # Right Householder
        if k < n
            G = Matrix(I, n, n)
            G_k = householder(B[k, k+1:end]')
            G[k+1:end, k+1:end] = G_k
            B = B * G
            V = V * G
        end
    end

    return U, B, V
end

function elementary_symmetric(λ::Vector{Float64})
    n = length(λ)
    s = zeros(Float64, n+1)   # s[0] to s[n]
    s[1] = 1.0                # s₀ = 1 (Julia is 1-indexed)

    for i in 1:n
        # Update from largest k downward to avoid overwriting s[k-1] too early
        for k in reverse(1:i)
            s[k+1] = s[k+1] + λ[i] * s[k]
        end
    end

    return s[2:end]  # s₁ to sₙ (since s[1] = s₀)
end

function determinstic_cross_decomposition(A::Matrix{Float64}, k:: Int)
    for iteration in 1:k
        B = copy(A)
        m, n = size(A)
        U, S, V = svd(A)
        I = []
        J = []
        minratio = Inf
        for i in 1:m 
            for j in 1:n 
                if B[i,j] == 0
                    throw("B[i,j] is equal to 0, it's time to implemnt pivoting")
                end
                x = S * transpose(V[j,:])
                y = (B[i,j])^(-1) * S * transpose(U[i,:])
                M = S - x * transpose(y)
                bidiag_matrix = golub_kahan_bidiagonalization(M)
                S = svd(bidiag_matrix)
                coefficients = elementary_symmetric(S)
                r = coefficients[m - k + iteration - 1] / coefficients[m - k + iteration]
                if r < minratio
                    i_t = i 
                    j_t = j
                end
            end
        end
    push!(I, i_t)
    push!(J, j_t)
    B = B - (B[:,j_t] * B[i_t, :] / B[i_t, j_t])
    end
end
