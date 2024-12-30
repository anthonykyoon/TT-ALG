using LinearAlgebra

function padding_tensor(tensor::Any, target_index::Int128)
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
    end 
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
    end
    #Calculating the first core
    tt_sum[1] = hcat(tt_a_interest, tt_b_interest)

    for tensor in 2:(d_a-1)
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
    end
    tt_sum[d_a] = vcat(tt_a_interest, tt_b_interest)
    return tt_sum
end 

function TT_SVD_1(input_tensor::Any, error)
    #rank of the tensor
    d = ndims(input_tensor)
    #storing the tensor train 
    tt_train = Vector{Any}(undef, d)
    #Frobeius Norm 
    trunc_param = error / sqrt(d - 1 ) * sqrt(sum(abs2, input_tensor))
    #Copying the tensor over 
    W = copy(input_tensor)
    #dimension of each index
    n = size(input_tensor)
    #temporary ranks of the tensor train 
    r =  [i * 1 for i in n]
    #SVD 
    for i in 2:d
        println(i)
        println(prod(size(W)))
        println(n[i] * r[i-1])
        W = reshape(W, Int(r[i-1] * n[i]), Int(prod(size(W))/(r[i-1] * n[i])))
        U, S, V = svd(W)
        # #truncation step 
        cumsum_singular = cumsum(S.^2)
        singular_squared = sum(S.^2)

        cutoff = findfirst(cumsum_singular.> (singular_squared - trunc_param^2))
        
        if cutoff === nothing
            throw("there exists no cutoff")
        end 

        U_trunc = U[:, 1:cutoff]
        S_trunc = diagm(S[1:cutoff])
        V_trunc = V[1:cutoff, :]
        r[i] = rank(U_trunc * S_trunc * V_trunc)
        println(r[i])
        tt_train[i] = reshape(U_trunc,( Int(r[i-1]), Int(n[i]), Int(r[i])))    
        W = S_trunc * (V_trunc)
    end
    tt_train[d] = W
    return tt_train 
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


    #Q and R can be computed by the rehspaingof the tensor G_k by reshaping it into r_{k-1} times n_k r_k
    for i in range(d, 2, -1)
        dim_current_core = size(G[i])
        Q, R = qr(reshape(G[i], dim_current_core[1], (dim_current_core[2] * dim_current_core[3])))

    end

end 
