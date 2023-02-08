### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ b5ce974f-d6b9-47b9-a284-cc012f2417ab
using Pkg

# ╔═╡ e8418bd9-6751-4aed-a809-c920c357a712
Pkg.develop(path="./VaeModels.jl")

# ╔═╡ c4d2dd7f-18a2-48f7-9c8f-03153f47e4a6
Pkg.add("BSON")

# ╔═╡ 0aca7f36-03af-445c-9204-76d493d69ce1
begin
using BSON
using Revise
using BenchmarkTools
end

# ╔═╡ a7d7a34f-695f-4402-b666-04b2c3c2ea27
using LinearAlgebra


# ╔═╡ 79348c4b-c5d0-43c8-b1ce-f7f8ca082231
using Zygote

# ╔═╡ 49f9ece9-7622-4356-b30a-94ede0e9f7a0
using CairoMakie:plot

# ╔═╡ ac588717-81cb-4eb6-b2e7-0bf273da43d1


# ╔═╡ 633fad74-7295-4e50-aad4-138fa7294a14
sum = 1

# ╔═╡ 2152a0a1-db56-4670-bc99-760b16d02f8a
x = 2

# ╔═╡ adc18a10-bfcf-4f0e-8942-34c2db790f51
n=5

# ╔═╡ cc7958b5-be46-4cbc-bda4-7a7e6a5bdee2
a = [0 0 0
    4 0 0
    0.1 6 0]

# ╔═╡ ad858b0c-11a9-413f-8c20-b21a710ca776
A = a - a'

# ╔═╡ 62cd26d0-9677-4d4f-9a19-04df17e47223
function myexp(x, n = 20)
	sum = one(x)
	theone = one(x)
	for i in n-1:-1:1
		sum = theone + x * sum / i;
	end
	sum
end

# ╔═╡ f53c5170-afb9-4f7e-8553-6545dead93b7


# ╔═╡ 43f127b0-61c7-4417-999c-02022969b502


# ╔═╡ fc9e0f4c-a3c9-4ca6-b289-9f0f8383affa
gradient(A -> det(exp(A - A')), A)

# ╔═╡ 1e536a82-0c9c-4842-9d88-1663ebbfacc2
max(abs.(eigvals(myexp(A))))

# ╔═╡ 71e9a3e1-4787-4ff0-883f-a9799236e403


# ╔═╡ eab1b8b6-888a-4747-aeb5-0e827f4f9b9f


# ╔═╡ 69a2030c-2036-4796-b077-ccb1e6849e64
@btime myexp(A)

# ╔═╡ 3e4bb808-4fa9-4e63-b6b5-7d30dee0f48f
@btime exp(A)

# ╔═╡ 811bebac-2ce7-4123-906a-fa8dbc610568


# ╔═╡ Cell order:
# ╠═0aca7f36-03af-445c-9204-76d493d69ce1
# ╠═b5ce974f-d6b9-47b9-a284-cc012f2417ab
# ╠═e8418bd9-6751-4aed-a809-c920c357a712
# ╠═c4d2dd7f-18a2-48f7-9c8f-03153f47e4a6
# ╠═ac588717-81cb-4eb6-b2e7-0bf273da43d1
# ╠═633fad74-7295-4e50-aad4-138fa7294a14
# ╠═2152a0a1-db56-4670-bc99-760b16d02f8a
# ╠═adc18a10-bfcf-4f0e-8942-34c2db790f51
# ╠═cc7958b5-be46-4cbc-bda4-7a7e6a5bdee2
# ╠═ad858b0c-11a9-413f-8c20-b21a710ca776
# ╠═62cd26d0-9677-4d4f-9a19-04df17e47223
# ╠═f53c5170-afb9-4f7e-8553-6545dead93b7
# ╠═43f127b0-61c7-4417-999c-02022969b502
# ╠═a7d7a34f-695f-4402-b666-04b2c3c2ea27
# ╠═79348c4b-c5d0-43c8-b1ce-f7f8ca082231
# ╠═fc9e0f4c-a3c9-4ca6-b289-9f0f8383affa
# ╠═1e536a82-0c9c-4842-9d88-1663ebbfacc2
# ╠═49f9ece9-7622-4356-b30a-94ede0e9f7a0
# ╠═71e9a3e1-4787-4ff0-883f-a9799236e403
# ╠═eab1b8b6-888a-4747-aeb5-0e827f4f9b9f
# ╠═69a2030c-2036-4796-b077-ccb1e6849e64
# ╠═3e4bb808-4fa9-4e63-b6b5-7d30dee0f48f
# ╠═811bebac-2ce7-4123-906a-fa8dbc610568
