using LinearAlgebra
using Maxvol 
using Random 
include("tt-algos.jl")

#there are two possiblities on this, we can either: Given tensor, we can decompose INTO tt 
#given a TT, we take lwo rank approximations of each node of the tensor train

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
    """
    dim_matrix = size(inmatrix)
    
    column_index = shuffle(1:dim_matrix[2])[1:r]
    row_index = shuffle(1:dim_matrix[1])[1:r]
    # println(column_index)
    # println(row_index)
    sub_matrix = copy(inmatrix[row_index, column_index])

    if det(sub_matrix) == 0
        return random_sub_matrix(inmatrix, r)
    else
        return sub_matrix, column_index, row_index
    end
end


function maxvol_square(inmatrix, tolerance, r)
    """
    require an n x m matrix M. r times r submatrix A_o with det(A_o) neq 0
    tolerance > 0, l = 0, b_ij = inftyand A_l =A_o 

    Ensure: A_l is a close to a dominant submatriux of M with index set (I_L, J_L)

    Args: 
        Tolerance: Error that we want 
        r: dimenison of the Submatrix
        l = 0 
    """
    @assert tolerance > 0
    @assert r > 0 

    #intializing the submatrix
    A_0, J_l, I_l = random_sub_matrix(inmatrix, r)
    #intializing the list of submatrices. 
    # A_l = []
    # push!(A_l, A_0)

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
            i = coordinates[1]
            j = coordinates[2]

            #replacement operation
            col = copy(inmatrix[I_l, j])
            A_0[:, i] = col
        end
        if !(any(x -> x, B_l.>(1+ tolerance)) & any(x -> x, C_l.>(1+tolerance)))
            condition == true
        end
    end
    return A_0
end



