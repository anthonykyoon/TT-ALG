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

function formrot(a,b)
    if abs(a) >= abs(b) 
        r = b / a 
        factor = sqrt(1 + r^2)
        a = abs(a) * factor
        c = factor^(-1)
        s = r * c
    else
        r = a / b 
        factor = sqrt(1 + r^2)
        b = abs(b) * factor 
        s = factor^(-1)
        c = r * s
    end
    return [[a,b], [c,s], [c , -s ; s, c]]
end

function applyrot(x,y,c,s,n:: Int)
    @assert length(x) >= n && length(y) >= n
    temp = c * x[1:n] + s * y(1:n)
    y[1:n] = -s * x[1:n] + c * y(1:n)
    x[1:n] = temp
    return [x, y]
end

function forchase(gamma, phi, z, n)
    cn, sn = formrot(z[n-1], z[n])[2]
    e = -sn * gamma[n-1]
    gamma[n-1] = cn * gamma[n - 1]
    phi[n-1] = sn * gamma[n]
    gamma[n] = cn * gamma[n]
    cn, sn = formrot(gamma[n], e)[2]
    result_applyrot = applyrot(phi[n-1], gamma[n-1], cn, sn, 1)
end


