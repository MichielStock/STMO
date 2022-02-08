### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 25700c98-eebd-11ea-17eb-4398f75594e0
using Plots, Combinatorics, PlutoUI

# ╔═╡ 2f8b1a76-8247-48b2-b324-4550b7bda94b
md"""
# Heuristics and metaheuristics

*STMO*

**Michiel Stock**

![](https://github.com/MichielStock/STMO/blob/master/chapters/10.Metaheuristics/Figures/logo.png?raw=true)

In this chapter, we will explore some more general algorithms to solve hard problems. We will start with local search and end up with some simple, though powerful metaheuristics. Since the algorithms reuse many components or can be abstracted for multiple problems, we will also illustrate Julia's dispatch system to design flexible software.
"""

# ╔═╡ c637a44c-eebd-11ea-3d4f-9539b52a7b88
md"""
## The knapsack problem revisited

To illustrate our methods, we will use a slightly larger instance of the knapsack problem. It has 19 items, making it just about feasible to go over all the combinations exhaustively but interesting enough to perform a more intelligent search.
"""

# ╔═╡ a912a3b8-a175-4606-bd49-7db772d46eeb
md"For the objective, we just yield the value of the total weight does not exceed the capacity, elsewise, we output a large negative number."

# ╔═╡ 15d1edbd-35f9-4e77-93ec-4302c9a34e5b
md"""
## Local search

Just about any local search algorithm can be summarized by the following template code.
"""

# ╔═╡ c9d7857b-6950-45c1-88f4-ac9c1552d194
md"""
Here, we have to specify:
- `f` : the objective function (we now maximize);
- `s₀` : the initial solution to start from;
- `tracker` : an object to track the progress, does not do anything by default;
- `N` : a function that defines the neighborhood;
- `S` : a way of selecting the next solution, determines the behaviour of `choose_next`
"""

# ╔═╡ f78f9bc8-1b59-4242-aefd-01d1758237cf
md"Below, we have defined two abstract types, `Neighborhood` and `Selector`. By means of type-based dispatch, the behaviour of the local search is modified. Using the function subtypes, we can access which types of neighborhoods and selectors we have provided."

# ╔═╡ 51db9a9b-ba03-470e-9acd-55f5dfadf702
md"## Hill climbing

Hill climbing is basically local search where one scans the neighborhood of a solution and picks the best one to proceed. The algorithm terminates when the solution can no longer be improved. 
"

# ╔═╡ 48ce73c5-cebf-4767-bfe8-019b116263e0
md"""
## Simulated annealing

Every step of the search, a random solution is sampled from the neighborhood. This new solution is accepted with a probability determined by its change in objective value and a temperature that is gradually decreased while running the algorithm.
"""


# ╔═╡ cc1042c5-f870-4c3d-8df7-9d36d6f9116a
@bind logTmin Slider(-4:1.0, show_value=true, default=-1)

# ╔═╡ 0c730376-cea1-447b-9285-179903c1689c
@bind logTmax Slider(1:5.0, show_value=true, default=2)

# ╔═╡ 4ee9f5c4-0e57-4c0d-a800-dd6d2b0d08b5
@bind r Slider(0.01:0.01:0.99, show_value=true, default=0.8)

# ╔═╡ 1520c3ea-bb12-489e-bb77-2de1b2249d8f
@bind kT Slider(1:20:1000, show_value=true, default=10)

# ╔═╡ d4d52b69-c7df-4437-a2a2-81917848e28c
md"""
## Tabu search

Tabu search explores a solution's neighborhood similarly to Hill climbing. The big difference is that when a modification is done, tabu search 'taboos' that change for a certain number of steps (determined by `tabu_length`). This forces the algorithm to explore regions where the objective deteriorates, potentially escaping local minima. 

Much flexibility is possible to design a tabu search. Here, when an item is added or removed from the knapsack, it is tabooed for a given number of iterations.
"""

# ╔═╡ 678756b9-7989-4542-ae4a-98744e84cc03
@bind tabu_length Slider(0:50, show_value=true, default=3)

# ╔═╡ e4b6996e-eebf-11ea-3f8d-a13ddf340bbf
md"## Neighborhoods

The neighborhood defines how we can hop from one solution to the next. For our algorithms, we have to implement an iterator over all the neighbors and a function to sample a random neighbor.

Since the knapsack problem works on strings, we consider the neighborhood with one or two bits flipped, the latter implying a much larger neighborhood.
"

# ╔═╡ 0da780f6-866b-48fd-976e-58962408dbb4
abstract type Neighborhood end

# ╔═╡ 9f990a70-be16-4f18-b4ac-c888e3461459
subtypes(Neighborhood)

# ╔═╡ 4cec040f-296d-4ffa-bab7-7540a5c5d412
struct OneFlip <: Neighborhood end

# ╔═╡ 252dedb6-c8a8-409e-b8e0-2f682f12c82c
struct TwoFlip <: Neighborhood end

# ╔═╡ 2929342c-6c6d-4f88-9d93-0a1057b6f031
# simple function that flips a bit in s at postion i
function flip!(s, i)
	s[i] ⊻= true
	return s
end

# ╔═╡ a635482d-7960-48f4-9af7-630af84b453d
flip(s, i) = flip!(copy(s), i)

# ╔═╡ cf01a5f0-c118-4ef8-934f-09b50d13de99
neighbors(s, ::OneFlip) = (flip(s, i) for i in 1:length(s))

# ╔═╡ 0180196a-88ab-4fc1-9878-0438fe8f61bb
neighbors(s, ::TwoFlip) = (flip!(flip(s, i), j) for i in 1:length(s) for j in 1:length(s) if i!=j)

# ╔═╡ 636ab8ac-b59d-493f-9c4d-3ddd5ec13a78
Base.rand(s, ::OneFlip) = flip(s, rand(1:length(s)))

# ╔═╡ afdddd5c-3558-4f63-80f9-5a05e33526ce
Base.rand(s, ::TwoFlip) = flip!(flip(s, rand(1:length(s))), rand(1:length(s)))

# ╔═╡ fcef48fe-eec3-11ea-22c2-0d17b1ee9af5
md"""## Selectors

Selectors characterize a local search method by changing the behavior of `choose_next`, which yield the next solution in the searching procedure.
"""

# ╔═╡ 0492484a-eec4-11ea-091f-5146073f6824
abstract type Selector end

# ╔═╡ 8ddfd6c3-3d93-4e60-8f84-52f7df080057
subtypes(Selector)

# ╔═╡ 326afda2-5de7-4720-82c1-38c0ae8637b2
struct BestNeighbor <: Selector end

# ╔═╡ d9734596-5f7e-48b7-888e-f8a20ae83062
struct FirstNeighbor <: Selector end

# ╔═╡ 7e25e4b7-cbc8-42ad-8dc4-e6f5720f3b6e
struct RandomImprovement <: Selector end

# ╔═╡ a696ac0f-f8b4-4215-bd0b-9c8123ee7260
struct Metropolis <: Selector
		T::Float64  # temperature parameter
	end

# ╔═╡ fad941eb-e986-4761-bf4a-eb8e9a81a85b
# loop over all neighbors and pick the best
function choose_next(f, s, N::Neighborhood, S::BestNeighbor)
	# current objective
	obj = f(s)
	for sn in neighbors(s, N)
		obj, s = max((obj, s), (f(sn), sn))
	end
	return s
end

# ╔═╡ a01d81c3-18ec-4360-be40-f4ffd4aeede9
# pick first neighbor that improves the objective
function choose_next(f, s, N::Neighborhood, S::FirstNeighbor)
	# current objective
	obj = f(s)
	for sn in neighbors(s, N)
		f(sn) > obj && return sn
	end
	return s
end

# ╔═╡ 43cabb77-3f88-4af0-94af-ccba0b0ba318
# pick a random neighbor and select it if it improves
function choose_next(f, s, N::Neighborhood, S::RandomImprovement)
	# current objective
	obj = f(s)
	sn = rand(s, N)
	if f(sn) > obj
		return sn
	else
		return s
	end
end

# ╔═╡ 3c35e7ea-0b43-405b-bfd4-3d661dedecba
# pick a random neighbor and select if it satisfies the Metropolis criterion
function choose_next(f, s, N::Neighborhood, S::Metropolis)
	# current objective
	obj = f(s)
	sn = rand(s, N)
	obj_sn = f(sn)
	if obj_sn > obj || rand() < exp(-(obj - obj_sn) / S.T)
		return sn
	else
		return s
	end
end

# ╔═╡ 0e46db27-3714-4889-af23-7a70a67dfe91
md"""
## Tracker

A tracker is a data structure to keep track of the search during the run of the algorithm.
"""

# ╔═╡ 8666d19e-eebd-11ea-11ad-93097ff088f1
abstract type Tracker end

# ╔═╡ 8bcf1d93-b396-45c7-a58c-fb01bc56de54
struct NoTracking <: Tracker end

# ╔═╡ 58664797-9c85-4bab-8c82-3bb743509caf
notrack = NoTracking()

# ╔═╡ 03d55585-7ed1-4916-b1a7-ec31fff1abb1
struct TrackSolutions{T} <: Tracker
		solutions::Vector{T}
		TrackSolutions(s) = new{typeof(s)}([])
	end

# ╔═╡ 95c0196c-faec-45e7-996c-2dc4de617bcd
struct TrackObj{T} <: Tracker
		objectives::Vector{T}
		TrackObj(T::Type=Float64) = new{T}([])
	end

# ╔═╡ cddcbed2-eff5-4288-b625-3dac12e9f1c1
track!(::NoTracking, f, s) = nothing

# ╔═╡ fdf3c648-e3a6-47ee-8171-a05a7f932d8c
track!(tracker::TrackSolutions, f, s) = push!(tracker.solutions, s)

# ╔═╡ 5d9f8c95-b4e3-478b-8b1c-21f31f2fcfe0
track!(tracker::TrackObj, f, s) = push!(tracker.objectives, f(s))

# ╔═╡ 3cd3fbba-eebd-11ea-164f-b39f495dea6e
function local_search(f, s₀, N::Neighborhood,
						S::Selector,
						tracker::Tracker=notrack;
						niter=10_000)
	s = s₀
	track!(tracker, f, s)
	for i in 1:niter
		s = choose_next(f, s, N, S)
		track!(tracker, f, s)
	end
	return s
end

# ╔═╡ 83429b08-d608-4f44-9ca6-67175f0df26b
function hill_climbing(f, s₀, N::Neighborhood,
					tracker=notrack;
					maxiter=10_000)
	s = s₀
	obj = f(s)
	track!(tracker, f, s)
	for i in 1:maxiter
		improved = false
		# search all the neighboring solutions of s
		for sn in neighbors(s, N)
			obj_sn = f(sn)
			if obj_sn > obj
				s = sn
				obj = obj_sn
				improved = true
			end
		end
		track!(tracker, f, s)
		# break if not improved
		!improved && break
	end
	return s
end	

# ╔═╡ 2541a5aa-21cf-41c3-b17d-7922cff10390
function simulated_annealing(f, s₀, N::Neighborhood, tracker=notrack;
				kT=100,  		# repetitions per temperature
				r=0.95,  		# cooling rate
				Tmax=1_000,     # maximal temperature to start
				Tmin=1)         # minimal temperature to end
	@assert 0 < Tmin < Tmax "Temperatures should be positive"
	@assert 0 < r < 1 "cooling rate is between 0 and 1"
	s = s₀
	obj = f(s)
	track!(tracker, f, s)
	# current temperature
	T = Tmax
	while T > Tmin
		# repeat kT times
		for _ in 1:kT
			sn = rand(s, N)  # random neighbor
			obj_sn = f(sn)
			# if the neighbor improves the solution, keep it
			# otherwise accept with a probability determined by the
			# Metropolis heuristic
			if obj_sn > obj || rand() < exp(-(obj-obj_sn)/T)
				s = sn
				obj = obj_sn
			end
		end
		track!(tracker, f, s)
		# decay temperature
		T *= r
	end
	return s
end
	

# ╔═╡ ca863fdb-cb23-4dcf-b9d7-3c0513fac13d
function tabu_search(f, s₀, N::Neighborhood, tracker=notrack;
			tabu_length=10, niter=100)
	s = s₀
	obj = f(s)
	track!(tracker, f, s)
	# this list keeps track of the items that are tabooed
	# one can only change an item in the knapsack it its tabu
	# value does not exceed the iteration number
	tabu_list = zeros(Int, length(s))
	snew = similar(s)
	for iter in 1:niter
		# we start with objective 0, because the objective can actively become worse
		obj = 0
		for sn in neighbors(s, N)
			# if any part of the neighbor is tabu, skip
			any(tabu_list[s.!=sn] .> iter) && continue
			obj_sn = f(sn)
			if obj_sn > obj
				snew .= sn
				obj = obj_sn
			end
		end
		tabu_list[s.!=snew] .= tabu_length + iter
		s .= snew
		track!(tracker, f, s)
	end
	return s
end	

# ╔═╡ ff39e011-7cbd-427a-8414-07cc1af885e0
md"## Appendix"

# ╔═╡ 9099c5e8-5110-4d32-9386-71bed0c7a495
md"Below are two examples of the knapsack problem: small one with 19 items and a big one with 300 items."

# ╔═╡ 13b06f4c-2408-4c5f-9604-9ad11caf0ed9
knapsack_string = """
19 31181
1945 4990
321 1142
2945 7390
4136 10372
1107 3114
1022 2744
1101 3102
2890 7280
962 2624
1060 3020
805 2310
689 2078
1513 3926
3878 9656
13504 32708
1865 4830
667 2034
1833 4766
16553 40006"""

# ╔═╡ b13c08a6-7209-4a9b-b1e4-b39569328a1b
md"below is a larger example you might use:"

# ╔═╡ a319305b-0632-4e9f-a378-6eadc761837a
knapsack_string2 = """
300 4040184
31860 76620
11884 28868
10492 25484
901 2502
43580 104660
9004 21908
6700 16500
29940 71980
7484 18268
5932 14564
7900 19300
6564 16028
6596 16092
8172 19844
5324 13148
8436 20572
7332 17964
6972 17044
7668 18636
6524 15948
6244 15388
635 1970
5396 13292
13596 32892
51188 122676
13684 33068
8596 20892
156840 375380
7900 19300
6460 15820
14132 34164
4980 12260
5216 12932
6276 15452
701 2102
3084 7868
6924 16948
5500 13500
3148 7996
47844 114788
226844 542788
25748 61996
7012 17124
3440 8580
15580 37660
314 1128
2852 7204
15500 37500
9348 22796
17768 42836
16396 39692
16540 39980
395124 944948
10196 24692
6652 16204
4848 11996
74372 178244
4556 11212
4900 12100
3508 8716
3820 9540
5460 13420
16564 40028
3896 9692
3832 9564
9012 21924
4428 10956
57796 138492
12052 29204
7052 17204
85864 205628
5068 12436
10484 25468
4516 11132
3620 9140
18052 43604
21 542
15804 38108
19020 45940
170844 408788
3732 9364
2920 7340
4120 10340
6828 16756
26252 63204
11676 28252
19916 47932
65488 156876
7172 17644
3772 9444
132868 318036
8332 20364
5308 13116
3780 9460
5208 12916
56788 136076
7172 17644
7868 19236
31412 75524
9252 22604
12276 29652
3712 9324
4516 11132
105876 253452
20084 48468
11492 27884
49092 117684
83452 199804
71372 171044
66572 159644
25268 60836
64292 154084
21228 51156
16812 40524
19260 46420
7740 18980
5632 13964
3256 8212
15580 37660
4824 11948
59700 143100
14500 35100
7208 17716
6028 14756
75716 181332
22364 53828
7636 18572
6444 15788
5192 12884
7388 18076
33156 79612
3032 7564
6628 16156
7036 17172
3200 8100
7300 17900
4452 11004
26364 63428
14036 33972
16932 40964
5788 14276
70476 168852
4552 11204
33980 81660
19300 46500
39628 95156
4484 11068
55044 131988
574 1848
29644 71188
9460 23020
106284 254468
304 1108
3580 8860
6308 15516
10492 25484
12820 31140
14436 34972
5044 12388
1155 3210
12468 30236
4380 10860
9876 24052
8752 21404
8676 21052
42848 102796
22844 54988
6244 15388
314 1128
314 1128
314 1128
314 1128
314 1128
314 1128
387480 926660
314 1128
314 1128
314 1128
314 1128
314 1128
15996 38692
8372 20444
65488 156876
304 1108
4756 11812
5012 12324
304 1108
314 1128
314 1128
314 1128
314 1128
314 1128
314 1128
314 1128
304 1108
1208 3316
47728 114556
314 1128
314 1128
314 1128
314 1128
314 1128
314 1128
104036 249172
5248 12996
312 1124
24468 58836
7716 18932
30180 72460
4824 11948
1120 3140
11496 27892
4916 12132
14428 34956
24948 59996
41100 98700
28692 69084
826 2352
3073 7846
7684 18868
5604 13708
17188 41476
34828 83756
7540 18380
8004 19508
2648 6796
5124 12748
3096 7892
166516 398532
13756 33212
9980 24260
15980 38660
9056 22012
5052 12404
8212 20124
11164 27028
13036 31572
23596 56892
2028 5156
7584 18468
5772 14244
4124 10348
5368 13236
4364 10828
5604 13708
8500 20700
7676 18652
8636 20972
4588 11276
4152 10404
4860 12020
5484 13468
8636 20972
5140 12780
236380 565460
116500 278900
36480 87660
16968 41036
5232 12964
13280 32060
138032 330364
9044 21988
22028 53156
4632 11564
13196 31892
65404 156708
28940 69580
865 2430
45988 110276
670 2040
4820 11940
41356 99212
39844 95588
897 2494
4028 9956
7924 19348
47756 114612
47036 112772
25908 62316
4516 11132
29460 70820
7964 19428
16964 41028
22196 53492
68140 163380
80924 193948
63700 152700
20860 50220
1682 4464
16804 40508
3195 8090
60348 144596
1901 4902
67468 161636
4772 11844
11196 27092
25836 62172
49676 119252
6188 15276
15588 37676"""

# ╔═╡ 14531290-8906-42ca-838c-aa52ebb91ca0
knapsack_data = split(knapsack_string, "\n") .|> s->(parse.(Int, split(s," ")))

# ╔═╡ 7cf27ad2-f7ee-41df-b917-8679f8eb5eda
const n_items, capacity = first(knapsack_data)

# ╔═╡ 1eeb0f50-8b9d-4260-a61a-f8e565eab32a
capacity

# ╔═╡ cf657065-fc1d-4c3d-b344-1ddfa2e620b5
md"A solution vector is just a binary vector of length $n_items, indicating whether to take an item or not. Let's start with an empty knapsack."

# ╔═╡ 0b574bcb-ab7e-42c8-8672-7dcafeb4d342
s₀ = zeros(Bool, n_items)

# ╔═╡ 710ddf4b-ef42-4143-98da-999e6acc5653
neighbors(s₀, OneFlip())  |> collect

# ╔═╡ 2eddc3c2-e6ac-49f6-a1f5-e4c6c55b37b2
neighbors(s₀, TwoFlip())  |> collect

# ╔═╡ 844b891c-d1bc-4d90-968f-e0a9459deb87
rand(s₀, OneFlip())

# ╔═╡ 15cc6750-54b1-46d4-8936-bb8f2b7375b2
rand(s₀, TwoFlip())

# ╔═╡ ee82356d-5100-4721-a7fe-bda834340a86
md"This problem only has 2^$n_items = $(2^n_items) combinations, feasible to enumerate:"

# ╔═╡ f60691ca-a055-404f-9557-3f1cc91345f6
const v = first.(knapsack_data[2:end])  # values

# ╔═╡ 222f0fa4-a043-44ee-b9a9-647f4bdd7a39
v # values of the items

# ╔═╡ 9ef3ad77-e19c-4509-9d8f-191c74c7ad40
const w = last.(knapsack_data[2:end])  # weights

# ╔═╡ 20a0cf1f-76cf-41d7-a8dd-a7eda3ec8a29
w # weight of the items

# ╔═╡ 24693db7-3cda-4fa3-a547-777b16f3993d
f_knapsack(s) = sum(w[s]) ≤ capacity ? sum(v[s]) : -10000

# ╔═╡ 46bb926b-ec86-4834-8d3b-4537e777aa73
f_knapsack(s₀)  # empty knapsack has a value of 0

# ╔═╡ c8eeac4b-6ab0-4976-b529-911ea6a65108
f_knapsack(ones(Bool, n_items))  # taking all the items results in a large negative cost

# ╔═╡ 62339637-415d-4d9e-9113-ebdfc85c3041
begin
	# stores the objective through the iterations
	local_tracker = TrackObj(Int) 

	s_local = local_search(f_knapsack, copy(s₀), 
			TwoFlip(),  # change me!
			RandomImprovement(),  # change me!
			local_tracker, niter=100)  # change the selector and neighborhood
end

# ╔═╡ 811337ab-970a-4cdc-90e7-ab04c5c96559
f_knapsack(s_local)

# ╔═╡ 3850f8bd-ac6f-4a62-8860-65941af59eae
begin
	# stores the objective through the iterations
	hc_tracker = TrackObj(Int) 

	s_hc = hill_climbing(f_knapsack, copy(s₀), OneFlip(),
			hc_tracker)
end

# ╔═╡ 70553739-f280-4972-8c4c-ecde6bd13e06
f_knapsack(s_hc)

# ╔═╡ 014738f4-3808-4875-b8f7-858ac2f1da6f
begin
	# stores the objective through the iterations
	sa_tracker = TrackObj(Int) 

	s_sa = simulated_annealing(f_knapsack, copy(s₀), OneFlip(),
			sa_tracker, Tmin=10^logTmin, Tmax=10^logTmax; r, kT)
end

# ╔═╡ ca99362b-b4ca-4c3f-9ea0-c223bd0b3cb5
f_knapsack(s_sa)

# ╔═╡ 79e6989f-32b6-4311-b375-010597999baf
begin
	# stores the objective through the iterations
	tabu_tracker = TrackObj(Int) 

	s_tabu = tabu_search(f_knapsack, copy(s₀), OneFlip(),
			tabu_tracker; tabu_length)	
end

# ╔═╡ 9788b851-c62f-4aa1-8ab4-cb42befd8612
f_knapsack(s_tabu)

# ╔═╡ 5870cf14-eebe-11ea-07cd-d51bcd1fa702
begin
	myblue = "#304da5"
	mygreen = "#2a9d8f"
	myyellow = "#e9c46a"
	myorange = "#f4a261"
	myred = "#e76f51"
	myblack = "#50514F"

	mycolors = [myblue, myred, mygreen, myorange, myyellow]
end;

# ╔═╡ 8bb6e5e4-eebd-11ea-2019-49875c096a68
scatter(w, v, xlabel="weight", ylabel="value", label="item", color=mygreen, legend=:bottomright, title="Items of the knapsack",yscale=:log, xscale=:log)

# ╔═╡ 73ca51f4-855c-41c5-a790-16edaa33415b
if n_items ≤ 20
	weights = Int[]
	values = Int[]

	for comb in combinations(1:n_items)
		push!(weights, sum(w[comb]))
		push!(values, sum(v[comb]))
	end
	best_obj = maximum(values[weights.≤capacity])
	scatter(weights[weights.≤capacity], values[weights.≤capacity], alpha=0.6, label="valid solutions", color=mygreen, xlabel="weight", ylabel="objective",
		legend=:bottomright, yscale=:log, xscale=:log)
	scatter!(weights[weights.>capacity], values[weights.>capacity], alpha=0.6, label="invalid solutions", color=myorange)
	vline!([capacity], label="capacity", color=myred, lw=2)
end

# ╔═╡ 607730d3-b7f4-44d9-a2f0-c3fff21ababc
best_obj  # best objective to be obtained

# ╔═╡ e294aea0-16c4-4523-b1fd-e55416c3e45c
Plots.plot(tracker::TrackObj; kwargs...) = plot(tracker.objectives, xlabel="iteratation", label="objective", lw=2, color=myred, legend=:bottomright; kwargs...)

# ╔═╡ 90baacdf-5fba-4e81-9b05-8c3dd6c70a6c
plot(local_tracker, title="Local search")

# ╔═╡ 8a3ea40f-865e-4415-a409-2e971344f31a
plot(hc_tracker, title="Hill climbing")

# ╔═╡ d595306b-1dc7-4d32-91bf-d0ffd65d380e
plot(sa_tracker, title="Simulated annealing")

# ╔═╡ 78f34919-c0ed-4111-884f-dc76ea5edac0
plot(tabu_tracker, title="Tabu search")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Combinatorics = "~1.0.2"
Plots = "~1.24.0"
PlutoUI = "~0.7.20"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "02a083caba3f73e42decb810b2e0740783022978"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.24.0"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "1e0cb51e0ccef0afc01aab41dc51a3e7f781e8cb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.20"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─2f8b1a76-8247-48b2-b324-4550b7bda94b
# ╠═25700c98-eebd-11ea-17eb-4398f75594e0
# ╟─c637a44c-eebd-11ea-3d4f-9539b52a7b88
# ╟─8bb6e5e4-eebd-11ea-2019-49875c096a68
# ╠═1eeb0f50-8b9d-4260-a61a-f8e565eab32a
# ╠═222f0fa4-a043-44ee-b9a9-647f4bdd7a39
# ╠═20a0cf1f-76cf-41d7-a8dd-a7eda3ec8a29
# ╟─cf657065-fc1d-4c3d-b344-1ddfa2e620b5
# ╠═0b574bcb-ab7e-42c8-8672-7dcafeb4d342
# ╟─a912a3b8-a175-4606-bd49-7db772d46eeb
# ╠═24693db7-3cda-4fa3-a547-777b16f3993d
# ╠═46bb926b-ec86-4834-8d3b-4537e777aa73
# ╠═c8eeac4b-6ab0-4976-b529-911ea6a65108
# ╟─ee82356d-5100-4721-a7fe-bda834340a86
# ╟─73ca51f4-855c-41c5-a790-16edaa33415b
# ╠═607730d3-b7f4-44d9-a2f0-c3fff21ababc
# ╟─15d1edbd-35f9-4e77-93ec-4302c9a34e5b
# ╠═3cd3fbba-eebd-11ea-164f-b39f495dea6e
# ╟─c9d7857b-6950-45c1-88f4-ac9c1552d194
# ╟─f78f9bc8-1b59-4242-aefd-01d1758237cf
# ╠═9f990a70-be16-4f18-b4ac-c888e3461459
# ╠═8ddfd6c3-3d93-4e60-8f84-52f7df080057
# ╠═62339637-415d-4d9e-9113-ebdfc85c3041
# ╠═811337ab-970a-4cdc-90e7-ab04c5c96559
# ╠═90baacdf-5fba-4e81-9b05-8c3dd6c70a6c
# ╟─51db9a9b-ba03-470e-9acd-55f5dfadf702
# ╠═83429b08-d608-4f44-9ca6-67175f0df26b
# ╠═3850f8bd-ac6f-4a62-8860-65941af59eae
# ╠═70553739-f280-4972-8c4c-ecde6bd13e06
# ╟─8a3ea40f-865e-4415-a409-2e971344f31a
# ╟─48ce73c5-cebf-4767-bfe8-019b116263e0
# ╠═2541a5aa-21cf-41c3-b17d-7922cff10390
# ╠═cc1042c5-f870-4c3d-8df7-9d36d6f9116a
# ╠═0c730376-cea1-447b-9285-179903c1689c
# ╠═4ee9f5c4-0e57-4c0d-a800-dd6d2b0d08b5
# ╠═1520c3ea-bb12-489e-bb77-2de1b2249d8f
# ╟─014738f4-3808-4875-b8f7-858ac2f1da6f
# ╠═ca99362b-b4ca-4c3f-9ea0-c223bd0b3cb5
# ╟─d595306b-1dc7-4d32-91bf-d0ffd65d380e
# ╟─d4d52b69-c7df-4437-a2a2-81917848e28c
# ╠═ca863fdb-cb23-4dcf-b9d7-3c0513fac13d
# ╠═678756b9-7989-4542-ae4a-98744e84cc03
# ╠═79e6989f-32b6-4311-b375-010597999baf
# ╠═9788b851-c62f-4aa1-8ab4-cb42befd8612
# ╟─78f34919-c0ed-4111-884f-dc76ea5edac0
# ╟─e4b6996e-eebf-11ea-3f8d-a13ddf340bbf
# ╠═0da780f6-866b-48fd-976e-58962408dbb4
# ╠═4cec040f-296d-4ffa-bab7-7540a5c5d412
# ╠═252dedb6-c8a8-409e-b8e0-2f682f12c82c
# ╠═2929342c-6c6d-4f88-9d93-0a1057b6f031
# ╠═a635482d-7960-48f4-9af7-630af84b453d
# ╠═cf01a5f0-c118-4ef8-934f-09b50d13de99
# ╠═0180196a-88ab-4fc1-9878-0438fe8f61bb
# ╠═710ddf4b-ef42-4143-98da-999e6acc5653
# ╠═2eddc3c2-e6ac-49f6-a1f5-e4c6c55b37b2
# ╠═636ab8ac-b59d-493f-9c4d-3ddd5ec13a78
# ╠═afdddd5c-3558-4f63-80f9-5a05e33526ce
# ╠═844b891c-d1bc-4d90-968f-e0a9459deb87
# ╠═15cc6750-54b1-46d4-8936-bb8f2b7375b2
# ╟─fcef48fe-eec3-11ea-22c2-0d17b1ee9af5
# ╠═0492484a-eec4-11ea-091f-5146073f6824
# ╠═326afda2-5de7-4720-82c1-38c0ae8637b2
# ╠═d9734596-5f7e-48b7-888e-f8a20ae83062
# ╠═7e25e4b7-cbc8-42ad-8dc4-e6f5720f3b6e
# ╠═a696ac0f-f8b4-4215-bd0b-9c8123ee7260
# ╠═fad941eb-e986-4761-bf4a-eb8e9a81a85b
# ╠═a01d81c3-18ec-4360-be40-f4ffd4aeede9
# ╠═43cabb77-3f88-4af0-94af-ccba0b0ba318
# ╠═3c35e7ea-0b43-405b-bfd4-3d661dedecba
# ╟─0e46db27-3714-4889-af23-7a70a67dfe91
# ╠═8666d19e-eebd-11ea-11ad-93097ff088f1
# ╠═8bcf1d93-b396-45c7-a58c-fb01bc56de54
# ╠═58664797-9c85-4bab-8c82-3bb743509caf
# ╠═03d55585-7ed1-4916-b1a7-ec31fff1abb1
# ╠═95c0196c-faec-45e7-996c-2dc4de617bcd
# ╠═cddcbed2-eff5-4288-b625-3dac12e9f1c1
# ╠═fdf3c648-e3a6-47ee-8171-a05a7f932d8c
# ╠═5d9f8c95-b4e3-478b-8b1c-21f31f2fcfe0
# ╠═e294aea0-16c4-4523-b1fd-e55416c3e45c
# ╟─ff39e011-7cbd-427a-8414-07cc1af885e0
# ╟─9099c5e8-5110-4d32-9386-71bed0c7a495
# ╠═13b06f4c-2408-4c5f-9604-9ad11caf0ed9
# ╟─b13c08a6-7209-4a9b-b1e4-b39569328a1b
# ╟─a319305b-0632-4e9f-a378-6eadc761837a
# ╠═14531290-8906-42ca-838c-aa52ebb91ca0
# ╠═7cf27ad2-f7ee-41df-b917-8679f8eb5eda
# ╠═f60691ca-a055-404f-9557-3f1cc91345f6
# ╠═9ef3ad77-e19c-4509-9d8f-191c74c7ad40
# ╟─5870cf14-eebe-11ea-07cd-d51bcd1fa702
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
