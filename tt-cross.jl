using LinearAlgebra
using Maxvol 
using Random 
include("tt-algos.jl")

#there are two possiblities on this, we can either: Given tensor, we can decompose INTO tt 
#given a TT, we take lwo rank approximations of each node of the tensor train

#Brainstorm 

#so i need to take a n-dimenionaal tensor and transofrm it into the tensor train. I should impolement the greedy max vol algo first. 

#n by r, do the rows r by n do the columsn n by n i have no idea


function random_sub_matrix(inmatrix, r)
    """
    Recursive function that will return a submatrix of r by r. 
    """
    dim_matrix = size(inmatrix)
    
    column_index = shuffle(1:dim_matrix[2])[1:r]
    row_index = shuffle(1:dim_matrix[1])[1:r]

    sub_matrix = deepcopy(inmatrix[column_index: row_index])

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
    A_l = []
    push!(A_l, A_0)

    #while loop to find the maxvol_sqaure
    condition = False
    l = 1
    while condition == False
        B_l = inmatrix[:, J_l] * inv(A_l[l])
    
        #checking if |b_{ij}| > 1 + tolerance
        if any(x -> x, B_l.>(1+ tolerance))
            bool_check = findall(x -> x > 1+ tolerance,  B_l)
            val_interest = max(B_l[bool_check])
            coordinates = findall(x -> x == val_interest, B_l)
            i = coordinates[1]
            j = coordinates[2] 
            
            #get the rows of each thingy 
            
        end

    end
end



