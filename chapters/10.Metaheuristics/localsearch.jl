### A Pluto.jl notebook ###
# v0.11.8

using Markdown
using InteractiveUtils

# ╔═╡ 25700c98-eebd-11ea-17eb-4398f75594e0
using Plots

# ╔═╡ 8666d19e-eebd-11ea-11ad-93097ff088f1
abstract type Tracker end

# ╔═╡ 8bb6e5e4-eebd-11ea-2019-49875c096a68


# ╔═╡ c637a44c-eebd-11ea-3d4f-9539b52a7b88


# ╔═╡ e4b6996e-eebf-11ea-3f8d-a13ddf340bbf
md"# Neighborhoods"

# ╔═╡ df0980b2-eec0-11ea-1775-e185922750ea
begin
	abstract type Neighborhood end
	
	struct OneFlip <: Neighborhood end
	
	oneflip = OneFlip()
	
	struct OneSwap <: Neighborhood end
	
	oneswap = OneSwap()
end

# ╔═╡ 47d523dc-eec1-11ea-2d55-e9c02f14cf4f
begin
	flip(s, i) = [j==i ? !sj : sj for (j, sj) in enumerate(s)]
	
	all_neighbors(s, ::OneFlip) = (flip(s, i) for i in 1:length(s))
	
	function swap(s, i, j)
		n = copy(s)
		n[i], n[j] = n[j], n[i]
		return n
	end
	
	all_neighbors(s, ::OneSwap) = (swap(s, i, j) for i in 1:length(s), j in 1:length(s) if s[i] != s[j])
end

# ╔═╡ 019334ee-eec2-11ea-2383-19f70f997887
all_neighbors([true, false, true, false], oneflip) |> collect

# ╔═╡ 9e624e5e-eec2-11ea-1aec-81cbeaee84c4
all_neighbors([true, false, true, false], oneswap) |> collect

# ╔═╡ 3f909330-eec3-11ea-1bd9-d7bb41cc052d
begin
	rand_neighbor(s, ::OneFlip) = flip(s, rand(1:length(s)))
		
	function rand_neighbor(s, ::OneSwap)
		i = rand(1:length(s))
		j = i
		while s[j] == s[i]
			j = rand(1:length(s))
		end
		return swap(s, i, j)
	end
end

# ╔═╡ c2abc1ea-eec3-11ea-3809-45bf92f5d133
rand_neighbor([true, false, true, false], oneflip)

# ╔═╡ caaf2bf2-eec3-11ea-1b01-9164c2d7d161
rand_neighbor([true, false, true, false], oneswap)

# ╔═╡ fcef48fe-eec3-11ea-22c2-0d17b1ee9af5
md"# Selectors"

# ╔═╡ 0492484a-eec4-11ea-091f-5146073f6824
abstract type Selector end

# ╔═╡ 1966a554-eec4-11ea-20d6-254b7898cf8d
begin
	
	struct BestNeighbor <: Selector end
	
	struct FirstNeighbor <: Selector end
	
	struct TwoStage <: Selector end
	
	struct RandomImprovement <: Selector end
	
	struct Metropolis <: Selector
		t
	end
end

# ╔═╡ b358ff22-eec4-11ea-2bbf-f12dbbf08c3a
begin
	function choose_next(f, s, N::Neighborhood, BestNeighbor)
		for i in 
	
end

# ╔═╡ e6c4a97a-eebe-11ea-259c-d3433ddd579d
begin
	struct NoTracking <: Tracker end
	
	struct SolStoring{T} <: Tracker
		solutions::Vector{T}
	end
	
	notrack = NoTracking()
end

# ╔═╡ 25968390-eebe-11ea-2c8b-8d472bca08c8
begin
	# don't track anything
	track(::NoTracking, s) = nothing
	
	# store solutons
	track(tracker::SolStoring, s) = push!(tracker.solutions, s)
end

# ╔═╡ 98eb82b0-eebd-11ea-1403-a56e2c9baed5
begin
	choose_next(f, s, N::Neighborhood, S::Selector) = s
	
end

# ╔═╡ 3cd3fbba-eebd-11ea-164f-b39f495dea6e
function local_search(f, s₀, N::Neighborhood,
						S::Selector,
						tracker::Tracker=notrack;
						max_iter=1000)
	s = s₀
	track(f, s)
	for i in 1:max_iter
		s = choose_next(f, s, N, S)
		track(f, s)
	end
	return s
end

# ╔═╡ 5870cf14-eebe-11ea-07cd-d51bcd1fa702


# ╔═╡ Cell order:
# ╠═25700c98-eebd-11ea-17eb-4398f75594e0
# ╠═8666d19e-eebd-11ea-11ad-93097ff088f1
# ╠═8bb6e5e4-eebd-11ea-2019-49875c096a68
# ╠═c637a44c-eebd-11ea-3d4f-9539b52a7b88
# ╠═3cd3fbba-eebd-11ea-164f-b39f495dea6e
# ╠═e4b6996e-eebf-11ea-3f8d-a13ddf340bbf
# ╠═df0980b2-eec0-11ea-1775-e185922750ea
# ╠═47d523dc-eec1-11ea-2d55-e9c02f14cf4f
# ╠═019334ee-eec2-11ea-2383-19f70f997887
# ╠═9e624e5e-eec2-11ea-1aec-81cbeaee84c4
# ╠═3f909330-eec3-11ea-1bd9-d7bb41cc052d
# ╠═c2abc1ea-eec3-11ea-3809-45bf92f5d133
# ╠═caaf2bf2-eec3-11ea-1b01-9164c2d7d161
# ╠═fcef48fe-eec3-11ea-22c2-0d17b1ee9af5
# ╠═0492484a-eec4-11ea-091f-5146073f6824
# ╠═1966a554-eec4-11ea-20d6-254b7898cf8d
# ╠═b358ff22-eec4-11ea-2bbf-f12dbbf08c3a
# ╠═25968390-eebe-11ea-2c8b-8d472bca08c8
# ╠═e6c4a97a-eebe-11ea-259c-d3433ddd579d
# ╠═98eb82b0-eebd-11ea-1403-a56e2c9baed5
# ╠═5870cf14-eebe-11ea-07cd-d51bcd1fa702
