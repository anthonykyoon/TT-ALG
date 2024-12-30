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

test_tensor = random_tensor(3)


@testset "Testing the SVD function" begin 
    @test TT_SVD_1( test_tensor,  0.00000000000001) != test_tensor
end 




