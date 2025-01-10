using LinearAlgebra
using OffsetArrays

#I didn't know how to deal with the data types of tensor trains, so I used the Any datatype

#helper functions
function mode_k_contraction(base, added, k::Int64)
    """
    Performs the mode k contraction

    Arguments:
        base: The tensor whose kth index will be replaced
        added: The sacrifical tensor who will be added into the tensor 
        k: The mode in which this will be applied 
    Returns:
        Tensor with the contraction completed 
    """
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

    Arguments:
        tt_train: Tensor Train 

    Returns: 
        first: Tensor derived from the tensor train 

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
        if size(first)[end] != size(interest_tensor)[1] 
            index1 = size(first)[end] 
            index2 = size(interest_tensor)[1]
            println("index 1 = $index1, index 2 = $index2 \n")
            throw("contraction cannot commence due to mismatch of indices")
        end
        first = first * interest_tensor
        updated_first_dims = size(first)
        first = reshape(first, Int(updated_first_dims[1] * interest_tensor_dims[2]), Int(updated_first_dims[2] / interest_tensor_dims[2]))
    end
    first = reshape(first, middle_indices...)
    @assert ndims(first) == d "the tensor indices do not align with the cores of the train"
    return first
end

function frobenius_tt_1(tt_train::Any)
    """
    Returns the frobenius norm of a tensor train
    Arguments:
        tt_train: A vector holding the tensor trains
    Returns:
        Frobenis Norm as a Float value
    """
    return sqrt(sum(abs2, tt_contraction(tt_train)))
end

function frobenius_tt(tt::Any) 
    """
    Alternative way of doing the frobenius norm

    Arguments: 
        tt: Tensor train 

    Returns:
        Frobenius norm as a Float
    """
    d = length(tt)
    # F is our accumulator for the partial inner product
    # Start with a 1×1 array with value 1.0
    F = [1.0]

    for k in 1:d
        core = tt[k]  
        (rL, n, rR) = size(core)
        
        F_new = zeros(rR, rR)
        
        # Contract over the n dimension
        for i in 1:n
            slice_i = @view core[:, i, :]  
            F_new .+= (slice_i') * F * slice_i
        end
        
        # Update F
        F = F_new
    end
    
    return sqrt(F[1, 1])  
end


function rq(A::Any)
    """
    Performs a QR decomposition on the rows

    Arguments:
        A: Matrix to be decomposed

    Returns: 
        Q = Transpose of the Q matrix such that the rows are orthonormal
        R = Transpose of the R matric such that R is upper triangular
    """
    F = qr(Transpose(A))
    R = Matrix(F.R)
   Q = Matrix(F.Q)
    return Transpose(Q), Transpose(R)
end


function padding_tensor(tensor::Any, target_index::Int128)
    """
    Pads a tensor for the use in the tt_direct_sum function

    Arguments:
        tensor: The tensor to be padded
        target_index: The axis in which we will pad in 
    
    Returns:
        padded_tensor: The padded tensor
    """
    rank_l, n, rank_r = size(tensor)
    
    if n == target_index || n > target_index 
        throw("Error, issue with the middle or target index")
    end

    padding = zeros(rank_l, (target_index - n), rank_r)
    padded_tensor = cat(tensor, padding, dim = 2)
    return padded_tensor
end

function trunc_svd(matrix::AbstractMatrix, trunc::Float64)
    """
    Performs the δ truncated SVD 

    Arguments:
        matrix: The matrix to be truncated
        trunc: The truncation parameter
    Returns:
        Utrunc: The trucnated U matrix
        Strunc: The truncated Σ matrix
        Vtrunc: The truncated V matrix    
    """
    U, S, V = svd(matrix)
    total_sqr = sum(S.^2)
    leftover = total_sqr
    V = transpose(V)
    k = 0 
    d = length(S)


    for i in 1:d
        leftover -= S[i]^2
        if leftover <= trunc^2
            k = i
            break
        end
    end
    if leftover > trunc^2
        k = d
    end 
    Utrunc = U[:, 1:k]
    Strunc = diagm(S[1:k])
    Vtrunc =  V[1:k, :]
    return Utrunc, Strunc, Vtrunc
end

#actual functions
function TT_Direct_Sum(tt_a, tt_b)
    """
    Takes two tensor trains and takes the direct sum of the two

    Arguments: 
        tt_a: first tensor train 
        tt_b: second tensor train 

    Returns: 
        tt_train: The two tensor trains summed together.
    """
    #Dimension of tt_a and tt_b
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
    tt_sum[1] = cat(tt_a_interest, tt_b_interest; dims=3)

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
    tt_sum[d_a] = cat(tt_a_interest, tt_b_interest; dims=1)
    return tt_sum
end 

function TT_SVD_1(input_tensor::Any, error)
    """
    Takes a tensor, and decomposes it into a tensor train of length d, where d is the rank of teh Tensor

    Arguments:
        input_tensor: the tensor to be decomposed into a tensor train 
        error: the maximum tolerated error

    Returns:
        tt_train: the resultant tensor train
    """
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
                throw("Something went wrong with the indices")
            else
                println("A severe error has occurred: $bad_case")
                throw(bad_case)
            end
        end
        if remaining_index == 0
            throw("index is somehow 0")
        end

        W = reshape(W, Int(r[i-1] * n[i]),  remaining_index)
        U_trunc, S_trunc, V_trunc = trunc_svd(W, trunc_param)
        
        r[i] = rank(U_trunc * S_trunc * (V_trunc))

        tt_train[i] = reshape(U_trunc,( Int(r[i-1]), Int(n[i]), Int(r[i])))    
        W = S_trunc * (V_trunc)
    end
    
    tt_train[d] = reshape(W, Int(r[d-1]), n[d], 1)
    return tt_train 
end


function TT_Round_1(tt_train, error::Float64)
    """
    Tensor Train Round algorithm, where we compress a tensor train 

    Arguements:
        tt_train: The tensor train to be compressed 
        error: the maximum error term allowed
    Returns:
        B: The compressed tensor train
    """
    d = length(tt_train)
    norm_tt = frobenius_tt(tt_train)
    trunc_param = error * norm_tt / sqrt(d-1)

    B = deepcopy(tt_train)

    for k in reverse(2:d)
        Gk = B[k]
        rk_1, nk, rk = size(Gk)
        folding_matrix = reshape(Gk, rk_1, nk * rk)
    
    
        Q, R = rq(folding_matrix)
        middle_index = size(Q, 1)
        B[k] = reshape(Transpose(Q), middle_index, nk, rk)
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
    #SVD 
    for k in 1:(d-1)
        Gk = B[k]
        rk_1, n_k, r_k = size(Gk)

        folding_matrix = reshape(Gk, rk_1 * n_k, r_k)

        #SVD
        Utrunc, Strunc, Vtrunc = trunc_svd(folding_matrix, trunc_param)

        newrank = size(Utrunc, 2)
        B[k] = reshape(Utrunc, rk_1, n_k, newrank)

        #storing result of S * v
        M = Strunc * (Vtrunc)

        #now contracting B_{k+1}
        Gnext = B[k+1]

        bk, n_k1, bk1 = size(Gnext)
        Gnextfolding = reshape(Gnext, bk, n_k1 * bk1)
        final_folding = M * Gnextfolding
        gamma = size(final_folding, 1)
        B[k+1] = reshape(final_folding, gamma, n_k1, bk1)
    end
    return B
end




