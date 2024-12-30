using LinearAlgebra 
using Test 
using Random
include("tt-algos.jl")


Random.seed!(1234)

function random_tensor(dimension::Int)
    dim_1 = 2
    dim_2 = 4
    dim_3 = rand(1:9)
    dim_4 = dimension
    return rand(dim_1, dim_2, dim_3, dim_4)
end

function random_tensor_train(ranks::Vector{Int}, dimension::Vector{Int})
    """ 
    Generate a tensor train
    Arguments:
        ranks: The ranks of each node, will check if the first and last ranks are 1
        dimensions: Vector with each seen index 
        
        Note that the length of ranks has to be one greater than that of dimensions 
    Returns:
        Tensor Train with length dimensionm with ranks
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
    for index in 1:length(dimension)
        tensor_train[index] = rand(ranks[index], dimension[index], ranks[index + 1])
    end 
    return tensor_train
end

#Building the test tensor and tensor trains
test_tensor = random_tensor(3)

test_rank = [1, 3, 2, 4, 1, 1]
test_dimension = [2, 1, 3, 2, 4]

test_tensor_train = random_tensor_train(test_rank, test_dimension)




@testset "testing the direct sum function" begin 
    test_rank = [1, 3, 2, 4, 1, 1]
    test_dimension = [2, 1, 3, 2, 4]
    
    @test length(test_tensor_train) == length(TT_Direct_Sum(test_tensor_train, test_tensor_train)) 
end 
    

println(TT_Direct_Sum(test_tensor_train, test_tensor_train))

# @testset "Testing the SVD function" begin 
#     @test TT_SVD_1( test_tensor,  0.00000001) != test_tensor
# end 




