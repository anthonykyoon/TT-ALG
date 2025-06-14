function cabs(x,y)
    if abs(x) >= abs(y)
        r = y / x 
        t = abs(x) * sqrt(1 + r^2)
    else 
        r = x / y 
        t = abs(y) * sqrt(1 + r^2)
    end
    return y 
end

function formrot(a, b)
    if abs(a) >= abs(b)
        r = b / a 
        factor = sqrt(1 + r^2)
        a_new = abs(a) * factor
        c = inv(factor)
        s = r * c
    else
        r = a / b 
        factor = sqrt(1 + r^2)
        a_new = abs(b) * factor 
        s = inv(factor)
        c = r * s
    end
    G = [c -s; s c]  # The rotation matrix
    return a_new, c, s, G
end


function applyrot(x, y, c, s, n::Int)
    @assert length(x) >= n && length(y) >= n
    temp = copy(c * x[1:n] + s * y[1:n])
    y[1:n] .= -s * x[1:n] .+ c * y[1:n]
    x[1:n] .= temp
    return nothing
end


function forchase(gamma, phi, z, n)
    _, cn, sn, _ = formrot(z[n-1], z[n])
    e = -sn * gamma[n-1]
    gamma[n-1] = cn * gamma[n - 1]
    phi[n-1] = sn * gamma[n]
    gamma[n] = cn * gamma[n]
    cn, sn = formrot(gamma[n], e)[2]
    applyrot(phi[n-1], gamma[n-1], cn, sn, 1)
    for i in reverse((n-2):1)
        _, cn, sn, _ = formrot(z[i], z[i+1])
        e = -sn * gamma[i]
        gamma[i] = cn * gamma[i]
        phi[i] = sn * gamma[i+1]
        gamma[i+1] = cn * gamma[i+1]
        d = cn * phi[i+1]
        phi[i+1] = sn * phi[i+1]
        _, cn, sn, _ =formrot(gamma[i], b)
        applyrot()
    end
end


