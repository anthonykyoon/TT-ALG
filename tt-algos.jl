using LinearAlgebra
using TSVD
using OffsetArrays


#helper functions
function mode_k_contraction(base, added, k::Int64)
    #performing the actual contraction itself
    base_perm = permutedims(base, [setdiff(1:ndims(base), [k])...])
    base_reshaped = reshape(base_perm, size(base)[k], : )
    tensor_interest = added * base_reshaped
    tensor_interest =  reshape(tensor_interest, (size(added, 1), size(base)[setdiff(1:ndims(base), [k])])...)
    #rearranging the indicies so that the tensor_interest has the same indices but k 
    indices = dims(base)
    indices[k] = size(added, 1)
    return reshape(tensor_interest, indicies)
end

function tt_contraction(tt_train::Any)
    """
    Make the tensor train back into a tensor. 
    """
    @assert length(size(tt_train[1])) == 3
    d = length(tt_train)

    #storing the middle indicies
    middle_indices = Vector{Any}(undef, d)
    dimensions_1 = size(tt_train[1])
    middle_indices[1] = dimensions_1[2]
    first = reshape(tt_train[1],dimensions_1[1]* dimensions_1[2], dimensions_1[3])
    for iteration in 2:d
        interest_tensor_dims = size(tt_train[iteration])
        middle_indices[iteration] = interest_tensor_dims[2]
        interest_tensor = reshape(tt_train[iteration], interest_tensor_dims[1], interest_tensor_dims[2] * interest_tensor_dims[3])
        @assert  size(first)[end] == size(interest_tensor)[1] "contraction cannot commence due to mismatch of indices"
        first = first * interest_tensor
        updated_first_dims = size(first)
        first = reshape(first, Int(updated_first_dims[1] * interest_tensor_dims[2]), Int(updated_first_dims[2] / interest_tensor_dims[2]))
    end
    first = reshape(first, middle_indices...)
    @assert ndims(first) == d "the tensor indices do not align with the cores of the train"
    return first
end

function rq(A::Any)
    F = qr(Transpose(A))
    R = Matrix(F.R)
   Q = Matrix(F.Q)
    return Transpose(Q), Transpose(R)
end


function padding_tensor(tensor::Any, target_index::Int128)
    rank_l, n, rank_r = size(tensor)
    
    if n == target_index || n > target_index 
        throw("Error, issue with the middle or target index")
    end

    padding = zeros(rank_l, (target_index - n), rank_r)
    padded_tensor = cat(tensor, padding, dim = 2)
    return padded_tensor
end


#actual functions
function TT_Direct_Sum(tt_a, tt_b)
    #Dimension of tt_a and TT)2
    d_a = length(tt_a)
    d_b = length(tt_b)
    
    if d_a != d_b
        throw("tensor trains are not the same length")
    end 
    #Storing the final TT Array 
    tt_sum = Vector{Array{Float64}}(undef,d_a)

    #Creating the new references for the 1st and last core
    tt_a_interest = copy(tt_a[1])
    tt_b_interest = copy(tt_b[1])

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
        tt_b_interest = copy(tt_b[tensor])

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
    tt_b_interest = copy(tt_b[d_b])

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

# function TT_SVD(input_tensor::Any, error::Any, r_max ::Any)
#     @assert r_max >=1 
#     d = ndims(input_tensor)
#     tt_train = Vector{Any}(undef, d)
#     n = size(input_tensor)
#     trunc_param = error * sqrt(sum(abs2, input_tensor)) / sqrt(d-1)
#     r = ones(Int128, d+1)
#     C = deepcopy(input_tensor)

#     for k in 2:(d)
#         C = reshape(C, Int(r[k-1] * n[k]), Int(prod(n)/(r[k-1] * n[k])))
#         U, S, V = svd(C)
#         cumsum_singular = cumsum(S.^2)
#         singular_squared = sum(S.^2)

#         cutoff = findlast(cumsum_singular.>=(singular_squared - trunc_param))
#         cutoff = min(r_max, cutoff)
#         Utrunc, Strunc, Vtrunc = tsvd(C, cutoff)
#         Strunc = Diagonal(Strunc)
#         r[k] = rank(Utrunc * Strunc * Vtrunc)
#         G[k] = reshape(Utrunc, Int(r[k-1]), n[k], Int(r[k]))
#         C = Strunc * VTrunc
#     end 
#     return tt_train
# end


function TT_SVD_1(input_tensor::Any, error)
    #rank of the tensor
    d = ndims(input_tensor)
    #storing the tensor train 
    tt_train = Vector{Any}(undef, d)
    #Frobeius Norm 
    frob_norm = sqrt(sum(abs2, input_tensor))
    trunc_param = error / sqrt(d - 1 ) * frob_norm
    #Copying the tensor over 
    W = copy(input_tensor)
    #dimension of each index
    n = size(input_tensor)
    #temporary ranks of the tensor train 
    r =  ones(d+1)
    r = OffsetArray(r, 0:(d))
    remaining_index = 0
    #SVD 
    #Use the reshaping matrix and dimension manipulation to do this. 

    for i in 1:d-1
        try 
            # Try regular calculation of remaining_index
            remaining_index = Int(prod(size(W)) / (r[i-1] * n[i]))
        catch bad_case 
            if bad_case isa InexactError
                println("Will start to pad.")
                
                next_index = cld(prod(size(W)), (r[i-1] * n[i]))
                amount_to_be_padded = (next_index * r[i-1] * n[i]) - prod(size(W))

                println("my math was wrong and I have no idea how to pad this properly")
                throw("Help please")
                # if mod(amount_to_be_padded, (r[i-1] * r[i])) == 0
                #     added_index = div(amount_to_be_padded, (r[i-1] * r[i]))
                #     padding = zeros(Int(r[i-1]), Int(added_index), r[i])
                #     W = cat(W, padding; dims=2)
                #      remaining_index = Int(prod(size(W)) / (r[i-1] * n[i]))
                #     println("Returning index from padding strategy 1.")
                # elseif mod(amount_to_be_padded, r[i] * n[i]) == 0
                #     added_index = div(amount_to_be_padded, (r[i] * n[i]))
                #     padding = zeros(added_index, n[i], r[i])
                #     W = cat(W, padding; dims=1)
                #      remaining_index = Int(prod(size(W)) / (r[i-1] * n[i]))
                #     println("Returning index from padding strategy 2.")
                # else
                #     println("Padding strategy failed.")
                #     println("iteration = $i")
                #     println(r[i-1])
                #     println(r[i])
                #     println(n[i])
                #     println(amount_to_be_padded)
                #     throw("Unable to resolve padding requirements.")
                # end
            else
                println("A severe error has occurred: $bad_case")
                throw(bad_case)
            end
        end
        if remaining_index == 0
            throw("index is somehow 0")
        end

        println("now reshaping")
        W = reshape(W, Int(r[i-1] * n[i]),  remaining_index)
        F = svd(W)
        U = F.U
        S = F.S
        V = F.V
        # #truncation step 
        cumsum_singular = cumsum(S.^2)
        singular_squared = sum(S.^2)

        cutoff = findlast(cumsum_singular.<= (singular_squared - trunc_param^2))
        println("cutoff = $cutoff for iteration $i")
        

        # if cutoff === nothing
        #     cutoff = 1
        # end 

        U_trunc = U[:, cutoff:end]
        S_trunc = diagm(S[cutoff:end])
        V_trunc = V[:, cutoff:end]
        println(size(U_trunc))
        println(size(S_trunc))
        println(size(V_trunc))
        r[i] = rank(U_trunc * S_trunc * Transpose(V_trunc))

        tt_train[i] = reshape(U_trunc,( Int(r[i-1]), Int(n[i]), Int(r[i])))    
        W = S_trunc * Transpose(V_trunc)
    end
    tt_train[d] = W
    return tt_train 
end

function TT_Round(input_tt:: Any, error_threshold::Float64)
    # #Store the tensor cores 
    # TT_cores = Vector{Any}(undef, d)
    #How long TT is 
    d = length(input_tt)
    #Copying the input_tt over
    G = copy(input_tt)
    #Rank of Each Core 
    #TODO fix the frobenius norm 
    tt_norm = sqrt(sum(abs2, tt_contraction(G)))
    #Truncation Parameter 
    trunc_param = (error_threshold) / (sqrt(d - 1)) * tt_norm


    #Q and R can be computed by the rehspaingof the tensor G_k by reshaping it into r_{k-1} times n_k r_k
    for i in d:2:(-1)
        dim_current_core = size(G[i])
        Q, R = qr(reshape(G[i], Int(dim_current_core[1]), Int(dim_current_core[2] * dim_current_core[3])))
        G[i] = Q
        #need to reshape this tensor 
        new_dim_core = size(G[i])
        G[i] = reshape(G[i], new_dim_core[1], new_dim_core[2], new_dim_core[3])
        G[i-1] = mode_k_contraction(G[i-1], R, 3)
    end

    for i in 1:(d-1)
        println("iteration $i")
        dim_current_core = size(G[i])
        reshaped_tensor = reshape(G[i], Int(dim_current_core[1]), Int(dim_current_core[2] * dim_current_core[3]))
        U, S, V = svd(reshaped_tensor)

        #truncation step 
        println(S)
        cumsum_singular = cumsum(S.^2)
        singular_squared = sum(S.^2)
        println(cumsum_singular)
        println(singular_squared)
        cutoff = findlast(cumsum_singular.<= (singular_squared - trunc_param))

        if cutoff === nothing 
            throw("cutoff does not exists")
        end

        G[i] = U[:, 1:cutoff]
        S = Diagonal(S[1:cutoff])
        V_Trunc = V[1:cutoff, :]
        G[i+1] = S * V_Trunc
    end
    return G
end 

function TT_Round_1(tt_train, error::Float64)
    """
    Second attempt at this. I'll add a nice docstring to this later. 
    """
    d = length(tt_train)
    norm_tt = sqrt(sum(abs2, tt_contraction(tt_train)))
    trunc_param = error * norm_tt / sqrt(d-1)

    B = deepcopy(tt_train)

    for k in reverse(2:d)
        Gk = B[k]
        rk_1, nk, rk = size(Gk)
        folding_matrix = reshape(Gk, rk_1, nk * rk)
    
    
        Q, R = rq(folding_matrix)
        middle_index = size(Q, 1)
        B[k] = reshape(Q, middle_index, nk, rk)

        # F  = qr(Transpose(folding_matrix))

        # Q = Matrix(F.Q)
        # R = F.R
        # #index caused by the qr decomp 
        # middle_index = size(Q, 2)

        # B[k] = reshape(Transpose(Q), middle_index, nk, rk)

        #contracting R into the matrix to the right of it 
        Gprev = B[k-1]
        p_rk_1, p_nk, p_rk = size(Gprev)
        #should match due to the mathematics of the QR decomposition

        Gprev_folding = reshape(Gprev, p_rk_1 * p_nk, p_rk )

        final_result =   Gprev_folding * R
        new_rk = size(final_result, 2)
        
        #reshaping the final result 
        B[k-1] = reshape(final_result, p_rk_1, p_nk, new_rk)
    end
    println("now doing the SVD")
    #SVD 
    for k in 1:(d-1)
        Gk = B[k]
        print("iteration is equal to $k")
        rk_1, n_k, r_k = size(Gk)

        folding_matrix = reshape(Gk, rk_1 * n_k, r_k)

        #SVD
        U, S, V = svd(folding_matrix)

        #truncation step 
        cumsum_singular = cumsum(S.^2)
        singular_squared = sum(S.^2)

        cutoff = findlast(cumsum_singular.<= (singular_squared - trunc_param^2))
        

        if cutoff === nothing
            println("reassigment of cutoff to 1")
            cutoff = 1 
        end 

        Utrunc = U[:, cutoff:end]
        Strunc = diagm(S[cutoff:end])
        Vtrunc = V[:, cutoff:end]
        #new core
        newramk = size(Utrunc, 2)
        B[k] = reshape(Utrunc, rk_1, n_k, newramk)

        #storing result of S * v
        M = Strunc * Transpose(Vtrunc)

        #now contracting B_{k+1}
        #TODO verify this step 
        Gnext = B[k+1]

        bk, n_k1, bk1 = size(Gnext)
        Gnextfolding = reshape(Gnext, bk, n_k1 * bk1)
        final_folding = M * Gnextfolding
        gamma = size(final_folding, 1)
        B[k+1] = reshape(final_folding, gamma, n_k1, bk1)
    end
    return B
end

