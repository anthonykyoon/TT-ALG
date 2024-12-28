using LinearAlgebra


function summing(limit::Float64, array_1: any)
    counter = 0
    for i in array_1
        if i != 0
            counter += 1
        elseif counter > limit | counter == limit
            return counter - i
        end
    end
    return counter     
end


function padding_tensor(tensor, target_index)
    rank_l, n, rank_r = size(tensor)
    
    if n == target_index || n > target_index 
        throw("Error, issue with the middle or target index")
    end

    padding = zeros(rank_l, (target_index - n), rank_r)
    padded_tensor = cat(tensor, padding; dim = 2)
    return padded_tensor
end


function TT_Direct_Sum(tt_a, tt_b)
    #Dimension of tt_a and TT)2
    d_a = length(tt_a)
    d_b = length(tt_b)
    
    if d_a != d_b
        throw("tensor trains are not the same length")

    #Storing the final TT Array 
    tt_sum = Vector{Array{Float64}}(n = d_a)

    #Creating the new references for the 1st and last core
    tt_a_interest = copy(tt_a[1])
    tt_b_interest = copt(tt_b[1])

    #Calculating the first core of the direct sum operation
    rank_1_l, n_1, rank_1_r = size(tt_a_interest)
    rank_2_l, n_2, rank_2_r = size(tt_b_interest)

    #Checking if things need to be padded 
    if  n_1 != n_2
        target_index = max(n_1, n_2)
        if n_1 > n_2
            tt_b_interest = padding_tensor(tt_b_interest, target_index)
        else
            tt_a_interest = padding_tensor(tt_a_interest, target_index)
    end
    #Calculating the first core
    tt_sum[1] = hcat(tt_a_interest, tt_b_interest)

    for tensor in range(2, (d_a -1), 1)
        #Retrieving information about the tensors copying for the case of padding 
        tt_a_interest = copy(tt_a[tensor])
        tt_b_interest = copt(tt_b[tensor])

        rank_1_l, n_1, rank_1_r = size(tt_a_interest)
        rank_2_l, n_2, rank_2_r = size(tt_b_interest)

        #Checking if n_a and n_b match. If not, pad the tensor with 0s to force hidden dimensions to match 
        if  n_1 != n_2
            target_index = max(n_1, n_2)
            if n_1 > n_2
                tt_b_interest = padding_tensor(tt_b_interest, target_index)
            else
                tt_a_interest = padding_tensor(tt_a_interest, target_index)
        end
        #Blank tensor 
        new_tensor = zeros(rank_2_l + rank_1_l, n_1 , rank_1_r + rank_2_r)

        #Performing the direct sum operation
        new_tensor[1:rank_1_l, :, 1:rank_1_r] = tt_a[tensor]
        new_tensor[(rank_1_l+1):end, :, (rank_1_r+1):end] = tt_b[tensor]
        tt_sum[tensor] = new_tensor
    end
    #Calculating the last tensor 
    tt_a_interest = copy(tt_a[d_a])
    tt_b_interest = copt(tt_b[d_b])

    #retrieving information 
    rank_1_l, n_1, rank_1_r = size(tt_a_interest)
    rank_2_l, n_2, rank_2_r = size(tt_b_interest)

    if  n_1 != n_2
        target_index = max(n_1, n_2)
        if n_1 > n_2
            tt_b_interest = padding_tensor(tt_b_interest, target_index)
        else
            tt_a_interest = padding_tensor(tt_a_interest, target_index)
    end

    tt_sum[d_a] = vcat(tt_a_interest, tt_b_interest)
    return tt_sum
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


        TT_cores[i] = reshape(V_trunc, r[i-1], n[i],r[i])

        n_bar = (n_bar * r[i -1]) /(n[i]r[i])

        W = U_trunc * S_trunc
    end
    TT_cores[1] = reshape(W, (1, n[1], r[1]))
    return TT_cores
end


 
function TT_Round(input_tt:: Array{Float64}, error_threshold::Float64)
    # #Store the tensor cores 
    # TT_cores = Vector{Any}(undef, d)
    #How long TT is 
    d = length(input_tt)
    #Copying the input_tt over
    G = copy(input_tt)
    #Rank of Each Core 
    tt_norm = norm(input_tt, "fro")
    #Truncation Parameter 
    trunc_parameter = (error_threshold) / (sqrt(d - 1)) * tt_norm
    #QR decomposition for each core
    for k in range(d, -1, 2)
        #TODO TODO TODO TODO TODO TODO double check if the functions actually match up 
        dimensions = size(G[k])
        rehape(G[k], dimensions[1], (dimensions[2] * dimensions[3]))
        G[k], R = qr(rehape(G[k], dimensions[1], (dimensions[2] * dimensions[3])))
        #TODO double check if this is the right contraction, intuitively this makes sense but I am not too sure on whether this is thr right contraction. Paper suggests a weird contraction
        G[k - 1] = G[k-1] *  R 
    end
    #Truncation via SVD 
    for k in range(1, 1, d-1)
        #TODO a little finicky on the indexing here, I think this is what it means to grab the middle index like this
        #Getting the middle index
        i_k =  size(input_tt[k])[2]
        dimensions = Dims(G[k])
        G[k], V, D = svd(reshape(G[k], dimensions[1] * i_k, dimensions[2] / i_k)) 
        G[k +1] = G[k +1] * transpose(D * V)
    end    
    #unfolding each matrix 
    for k in range(1, 1 , d)
        dimensions =  Dims(input_tt[k])
        G[k] = reshape(G[k], )
    end
end



# function TT_Round(input_tensor::Array{Float64}, error_threshold::Float64)
    
#     #Storing the Tensor cores
#     TT_cores = Vector{Any}(undef, d)

#     #Number of vector space in the Tensor
#     dimensions = ndims(input_tensor)

#     #Rank of each index
#     n = size(input_tensor)

#     #Frobeius norm
#     input_tensor_norm = norm(input_tensor, "fro")

#     #calculating the Truncation parameter 
#     trunc_param = (error_threshold / sqrt(dimensions - 1)) * input_tensor_norm

#     #Copying the input tensor 
#     W = copy(input_tensor)
#     for i in dimensions:-1:2
#         q_i , r_i = qr(reshape(W[i], :, size(W[i], 3)))
#         W[i] = q_i
#         W[k - 1] = W[k-1] * r_[i]
#     end
#     for k in 1:1:dimensions-1
#         W[k], S, V = svd(reshape(W[k], :, size(W[i], 3)))
#         #truncation step
#         rank_k = sum(S.>= trunc_param)
#     end
# end