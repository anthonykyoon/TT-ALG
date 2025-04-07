using LinearAlgebra
using OffsetArrays

# First attempt at the algorithm in Low-rank approximation in the Frobenius norm by column and row subset selection. 


#helper functions
function cabs(x, y)
    if abs(x) >= abs(y)
        r = y / x 
        t = abs(x) * sqrt(1 + r^2)
    else
        r = x / y 
        t = abs(y) * sqrt(1 + r^2)
    end
    return t
end

function formrot(a,b)
    if abs(x) >= abs(y)
        r = x / y 
        factor = sqtr(1 + r^2)
        c = 1 / factor
        s = r * c
    else
        r = x / y 
        factor = sqtr(1 + r^2)
        c = 1 / factor
        s = r * c
    end
    return c, s
end

function applyrot(x, y, c , s)
    #WIP 
end



# Main function
function cross_decomposition(matrix :: AbstractArray, k :: Int128)
    m, n = size(matrix)
    @assert k <= m 
    I = []
    J = []
    B = deepcopy(matrix)
    for i in 1:k 
        U, S, V = svd(B)
        min_ratio = +Inf
        for i in 1:m 
            for j in 1:n 
                x = S * V[j,:]'
                y = B[i,j]^(-1) * S * U(i,:)'


            end
        end
    end
end


