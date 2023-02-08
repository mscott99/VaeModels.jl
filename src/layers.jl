mutable struct orthonormallayer
    trimatrix
    filter
    fullmatrix
end
orthonormallayer() # lower triangular with a filter.


a = [0 0 0
    4 0 0
    0.1 6 0]
A = a - a'

A = [1.0 0 0
    0 2 0
    0 0 4]

exp(A)


@btime exp(A)

using LinearAlgebra
exp(a)


A = a
function expA(A)
    place = I(size(A, 1))
    n = 1
    count = zero(A)
    N = 1e3
    for i in 1:N
        count .= count .+ (1 ./ n) .* place
        #@show n place
        n *= i
        place = A * place
    end
    place
end

A

A = 2
lastterm = one(A)
acc = zero(A)
n = 1
count = zero(A)
N = 3

for i in 1:N
    acc += lastterm
    @show i
    @show lastterm * A
    lastterm = lastterm * (A / i)
end

acc
exp(2)



expA()
I(size(a, 1))



expA(a)
exp(a)

@btime qr(a)

