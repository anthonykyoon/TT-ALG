using LinearAlgebra 
using Test 
using Random
include("tt-algos.jl")
include("tt-cross.jl")


Random.seed!(123693456785743121212256782543256786528432145145722222212890765432567897651313814203)


function random_tensor(dimension::Int)
    """
    Generate a random tensor. 1st dimension = 1, 2nd dimension = 2
    3rd dimension = a random integer from 1 - 9 4th dimension = dimension

    Arguements:
        dimension: dimensions of the 4th indices
    Returns
        random tensor with aforementioned dimensions
    """
    dim_1 = 2
    dim_2 = 4
    dim_3 = rand(1:10)
    dim_4 = dimension
    return rand(dim_1, dim_2, dim_3, dim_4, 10) 
end

function random_tensor_train(ranks::Vector{Int}, dimension::Vector{Int})
    """ 
    Generate a tensor train
    Arguments:
        ranks: The ranks of each node, will check if the first and last ranks are 1
        dimensions: Vector with each seen index 
        
        Note that the length of ranks has to be one greater than that of dimensions 
    Returns:
        Tensor Train with length dimensions
    """
    d = length(dimension)
    if length(ranks) != d + 1
        throw("Invalid Inputs ")
    end

    if ranks[1] != 1 || ranks[end] != 1
        throw("Wrong rank vector")
    end
    #intializing the tensor train 
    tensor_train = Vector{Any}(undef, d)

    #storing the the tensors now 
    for index in 1:d
        tensor_train[index] = rand(ranks[index], dimension[index], ranks[index + 1])
    end 
    return tensor_train
end

function direct_sub(tt_a, tt_b)
    """
    Element wise subtraction between two tensor trains in the form tt_a - tt_b
    Arguements:
        tt_a: first tensor train
        tt_b: second tensor train 
    Returns:
        tt_sub: the tensor train resultant of elementwise subtraction
    """
    @assert length(tt_a) === length(tt_b) "tensor trains must have the same length"
    d = length(tt_a)
    tt_sub = Vector{Any}(undef, d)
    for i in 1:d
        @assert size(tt_a[i]) === size(tt_b[i]) "must have the same dimensions"
        c = tt_a[i].-tt_b[i]
        tt_sub[i] = c
    end
    return tt_sub
end


#Building the test tensor and tensor trains
test_tensor = random_tensor(3)
test_rank = [1, 3, 2, 4, 9, 1]
test_dimension = [3, 4, 3, 2, 4]
test_matrix = rand(Float64, (10, 15))


# test_tensor_train = random_tensor_train(test_rank, test_dimension)
# sub, coord = maxvol_square(test_matrix, 0.5, 6, true)
# println(test_matrix)
# println(sub)
# print(length(coord))

# cross = (TT_Cross_ACA(test_tensor, 0.000000001))
# println(frobenius_tt_1(test_tensor))
# println(frobenius_tt_1(cross))



# @testset "testing the direct sum function" begin 
#     test_rank = [1, 3, 2, 4, 1, 1]
#     test_dimension = [2, 1, 3, 2, 4]
    
#     @test length(test_tensor_train) == length(TT_Direct_Sum(test_tensor_train, test_tensor_train)) 
# end 
    

# println(TT_Direct_Sum(test_tensor_train, test_tensor_train))

# @testset "Testing the SVD function" begin 
#     @test TT_SVD_1(test_tensor,  0.0001) != test_tensor
# end 

# @testset "testing the round function" begin
#     @test length(test_tensor_train) == length(TT_Round_1(test_tensor_train, 0.00000000000000000001))
# end

tt_train = TT_SVD_1(test_tensor*10, 0.00000000001)
println(tt_train)

# tt2 = TT_Direct_Sum(tt_train, tt_train)
# tt_simp = TT_Round_1(tt2, 0.00000001, 3)
# tt_sub = direct_sub(tt_train, tt_simp)



# d = length(tt_simp)
# for i in 1:d
#     println("simp")
#     println(size(tt_simp[i]))
#     println("og")
#     println(size(tt_train[i]))
#     println("\n")
# end
# println("og frob norm = $(frobenius_tt_1(tt_train))")
# println("simp frob norm = $(frobenius_tt_1(tt_simp))")
# println("error term norm = $(frobenius_tt_1(tt_sub))")




