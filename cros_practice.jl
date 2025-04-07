using LinearAlgebra
using OffsetArrays

# First attempt at the algorithm in Low-rank approximation in the Frobenius norm by column and row subset selection. 


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
                
            end
        end
    end
end


