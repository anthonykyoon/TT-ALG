using LinearAlgebra
#TT-SVD


#I could have used cumsum here
function summing(limit:: Float64, array::Any)
    counter = 0
    for i in array
        if i != 0
            counter += i
            if counter > limit | counter == limit
                return counter - i
    return counter
end



function TT_SVD_1(input_tensor::Array{Float64}, error_threshold, r_max::int)
    #Storing the Tensor cores
    TT_cores = Vector{Any}(undef, d)
    
    #Number of vector spaces in the tensor 
    dimensions = ndims(input_tensor)
    
    #Rank of each index
    n = size(input_tensor)
    
    #Frobeius norm
    input_tensor_norm = norm(input_tensor, "fro")

    #Truncation parameter
    trunc = (error_threshold / sqrt(dimensions - 1)) * input_tensor_norm

    #n bar
    n_bar = [i * 1 for i in n ]

    r = ones(Int, dimensions)
    
    #Copying the results
    W = copy(input_tensor)
    
    for i in dimensions:-1:2
        W = reshape(W, (n_bar / (n[i]r[i]),n[i]r[i]))

        #SVD
        U, S, V = svd(W)

        eigen_squared = copy(S.^2)
        r_sigma = summing(array = eigen_squared, limit = trunc^2)
        r[i-1] = min(r_max, max(1, r_sigma))

        #Truncation
        U_trunc = U[:, 1:r[i-1]]
        V_trunc = V[1:r[i-1],:]
        S_trunc = Diagonal(S[1:r[i-1]])

        T[i] = reshape(V_trunc, r[i-1], n[i],r[i])

        n_bar = (n_bar * r[i -1]) /(n[i]r[i])

        W = U_trunc * S_trunc
    end
    TT_cores[1] = reshape(W, (1, n[1], r[1]))
    return TT_cores
end

function TT_Round(input_tensor::Array{Float64}, error_threshold::Float64)
    
    #Storing the Tensor cores
    TT_cores = Vector{Any}(undef, d)

    #Number of vector space in the Tensor
    dimensions = ndims(input_tensor)

    #Rank of each index
    n = size(input_tensor)

    #Frobeius norm
    input_tensor_norm = norm(input_tensor, "fro")

    #calculating the Truncation parameter 
    trunc_param = (error_threshold / sqrt(dimensions - 1)) * input_tensor_norm

    #Copying the input tensor 
    W = copy(input_tensor)
    for i = dimensions:-1:2
        q_i , r_i = qr(reshape(W[i], :, size(W[i], 3)))
        W[i] = q_i
        W[k - 1] = W[k-1] * r_[i]
    for k in 1:1:dimensions-1
        W[k], S, V = svd(reshape(W[k], :, size(W[i], 3)))
        #truncation step
        rank_k = sum(S.>= trunc_param)

        

    end
end