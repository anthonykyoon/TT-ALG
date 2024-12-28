using LinearAlgebra 

function random_tensor(dimension)
    dim_1 = 2
    dim_2 = 4
    dim_3 = rand(1:9)
    dim_4 = dimension
    return rand(dim_1, dim_2, dim_3, dim_4)
end



