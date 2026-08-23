using LinearAlgebra
using Maxvol 
using Random 
using OffsetArrays

#there are two possiblities on this, we can either: Given tensor, we can decompose INTO ACA 
#given a ACA, we take lwo rank approximations of each node of the tensor train

#Brainstorm 

#so i need to take a n-dimenionaal tensor and transofrm it into the tensor train. I should impolement the greedy max vol algo first. 

#n by r, do the rows r by n do the columsn n by n i have no idea

# function cart_to_tuple(cart :: CartesianIndex)
#     for i in CartesianIndex

#     end
    
# end


function random_sub_matrix(inmatrix, r)
    """
    Recursive function that will return a submatrix of r by r. 

    Arguements:
        inmatrix: Matrix to be broken down 
        r: size of submatrix
    
    Returns:
        sub_matrix: Actual sub_matrix
        column_index: list of indices of the column 
        row_index: list of the indices of the row
    """
    #Grabbing the dimensions of the matrix
    dim_matrix = size(inmatrix)

    #grabbing random indices to create the submatrix
    column_index = shuffle(1:dim_matrix[2])[1:r]
    row_index = shuffle(1:dim_matrix[1])[1:r]
    sub_matrix = copy(inmatrix[row_index, column_index])

    #checking if the matrix is invertible 
    if det(sub_matrix) == 0
        return random_sub_matrix(inmatrix, r)
    else
        return sub_matrix, column_index, row_index
    end
end



function maxvol_square(inmatrix, tolerance, r, ACA:: Bool)
    """
    require an n x m matrix M. r times r submatrix A_o with det(A_o) neq 0
    tolerance > 0, l = 0, b_ij = inftyand A_l =A_o 

    Ensure: A_l is a close to a dominant submatriux of M with index set (I_L, J_L)

    Args: 
        Tolerance: Error that we want 
        r: dimenison of the Submatrix
        ACA :: Bool : Returns the indices each column and row. 
    Returns:
        A_O: submatrix with largest in volume

    """
    @assert tolerance > 0
    @assert r > 0 

    #intializing the submatrix
    A_0, J_l, I_l = random_sub_matrix(inmatrix, r)
    #intializing the list of submatrices. 

    #while loop to find the maxvol_sqaure
    condition = false
    l = 1
    while condition == false
        B_l = inmatrix[:, J_l] * inv(A_0)
        #checking if |b_{ij}| > 1 + tolerance
        if any(x -> x, B_l.>(1+ tolerance))
            println("row operations")
            bool_check = findall(x -> x > 1+ tolerance,  B_l)
            val_interest = maximum(B_l[bool_check])
            coordinates = findall(x -> x == val_interest, B_l)
            println(coordinates)
            i = coordinates[1][1]
            j = coordinates[1][2] 
            
            #replacement operation 
            #ith row of M 
            row = copy(inmatrix[i, J_l])
            A_0[j, :] = row            #
        end
        #checking if |c_{ij}| > 1 + tolerance
        C_l = inv(A_0) * inmatrix[I_l, :]
        if any(x -> x, C_l.>(1+tolerance))
            println("column operations")
            bool_check = findall(x -> x > 1 + tolerance, C_l)
            val_interest = maximum(C_l[bool_check])
            coordinates = findall(x -> x == val_interest, C_l)
            i = coordinates[1][1]
            j = coordinates[1][2]

            #replacement operation
            col = copy(inmatrix[I_l, j])
            A_0[:, i] = col
        end
        if !(any(x -> x, B_l.>(1+ tolerance)) & any(x -> x, C_l.>(1+tolerance)))
            condition = true
        end
    end
    if ACA
        coordinates = findall(x -> x in A_0, inmatrix)
        return A_0, coordinates
    else
        return A_0
    end
end



function im_ACA(inmatrix, tolerance, sort = false)
    """
    Does the cross decomposition 
    Arguements:
        inmatrix: matrix to be decomposed
        tolerance: How much we consider the highest value
        sort:: Bool: Whether we return a sorted version of the indices or not. 
    Returns:
        I: Row indices
        J: Column indices 
    """
    #set R_0 = R. I = empty set, J = empty set 
    R_o = deepcopy(inmatrix)
    dimensions = size(R_o)
    I = []
    J = []
    in_fro = norm(inmatrix)
    condition = true
    while condition == true
        abs_R_o = abs.(R_o)
        cart_index = argmax(abs_R_o)
        i = cart_index[1]
        j = cart_index[2]
        push!(I, i)
        push!(J, j)
        delt = R_o[i,j]
        u_k = R_o[:, j]
        v_k = transpose(R_o[i, :])./ delt
        R_o = R_o - (u_k * v_k)
        if norm(R_o) <= tolerance * in_fro
            condition = false
        end
    end
    if sort
        return sort(I), sort(J)
    else
        return I, J
    end
end


function TT_Cross_ACA_1(inmatrix, tolerance)
    #storing the tensors
    tt_storage = []
    d = ndims(inmatrix)
    dims = size(inmatrix)
    W = deepcopy(inmatrix)
    #offsetting the array for ease of access 
    r = ones(d+1)
    r = OffsetArray(r, 0:d)
    for i in 1:d-1
        remaining_index = Int(prod(size(W)) / (r[i-1] * dims[i]))
        folding_matrix = reshape(W, Int(r[i-1]) * Int(dims[i]), remaining_index)

        #doing the cross decomposition 
        I, J = im_ACA(folding_matrix, tolerance)

        #Constructing the matrices 
        C = folding_matrix[:, J]
        R = folding_matrix[I, :]
        A = folding_matrix[I,J]

        #constructing and constructing the core. 
        r[i] = Int(size(A)[1])
        core = C * inv(A)
        core = reshape(core, Int(r[i-1]), Int(dims[i]), Int(r[i]))
        push!(tt_storage, core)
        W = R
    end
    #handling edge case
    dim_final = size(W)
    @assert length(dim_final) == 2
    W = reshape(W, dim_final[1], dim_final[2], 1)
    push!(tt_storage, W)
    return tt_storage
end


# function TT_Cross_ACA(inmatrix, tolerance)
#     #storing the cores 
#     tt_storage = []
#     d = ndims(inmatrix)
#     dims = size(inmatrix)
#     W = deepcopy(inmatrix)
#     r = ones(d+1)
#     for i in 1:d-1
#         println("iteration i is $i")
#         #begin reshaping the tensor 
#         remaining_index = Int(prod(size(W)) / dims[i])

#         folding_matrix = reshape(W, dims[i], remaining_index)
#         #doing the cross decomposition 
#         I, J = im_ACA(folding_matrix, tolerance)

#         #constructing the matrices
#         C = folding_matrix[:, J]
#         R = folding_matrix[I, :]
#         A = folding_matrix[I,J]

#         #creating and storing the cores
#         core = C * inv(A)
#         dims_core = size(core)
#         println("the size of the core is $dims_core")

#         #checking if it is the 1st core
#         if i == 1
#             core = reshape(core, 1, dims[i], dims_core[2])
#         else
#             println(dims[i])
#             previous_core = tt_storage[i - 1]
#             println((size(previous_core)[3]))

#             #TODO fix this logic error
#             core = reshape(core, size(previous_core)[3], dims[i], Int(prod(size(core)) / ((size(previous_core)[3]) * dims[i])))
#         end
#         push!(tt_storage, core)
#         W = R        
#     end
#     #last core
#     previous_core = tt_storage[i - 1]
#     final_core = reshape(W, size(previous_core)[end], dims[end], 1)
#     push!(tt_storage, final_core)
#     return tt_storage
# end
