#=
Created on Sunday 19 April 2020
Last update: Sunday 26 April 2020

@author: Michiel Stock
michielfmstock@gmail.com

Illustration of the algorithms for solving TSP.

Currently implements:
    - nearest neighbors
    - greedy
    - random insertion
    - hill climbing
    - simulated annealing
    - tabu search
=#

using STMO
using STMO.TSP

using Plots

tsp = totoro_tsp()
name = "totoro"

n = length(tsp)

# NEAREST NEIGHBORS

"""
    nearestneighbors(tsp::TravelingSalesmanProblem; start::Int)

Solves the TSP using the nearest neighbors. Provide a starting city using `start`.
If none is provide, a random one is chosen.
"""
function nearestneighbors(tsp::TravelingSalesmanProblem; start::Int)
    cities_to_add = Set(cities(tsp))
    delete!(cities_to_add, start)
    tour = [start]
    current = start
    cost = 0.0
    while length(cities_to_add) > 0
        # find closest city not in tour
        c, next = minimum((dist(tsp, current, n), n) for n in cities_to_add)
        push!(tour, next)
        delete!(cities_to_add, next)
        cost += c
        current = next
    end
    return tour, cost
end

nearestneighbors(tsp::TravelingSalesmanProblem) = nearestneighbors(tsp, start=rand(cities(tsp)))

"""
    bestnearestneighbors(tsp::TravelingSalesmanProblem;
                        ntry::Union{Nothing,Int}=nothing)

Chooses the best nearest neighbor solution over all cities. If `ntry` is provided,
a random number of cities is tried. This is done if searching all cities is too
involved.
"""
function bestnearestneighbors(tsp::TravelingSalesmanProblem;
                        ntry::Union{Nothing,Int}=nothing)
    if ntry isa Nothing
        cities_to_try = cities(tsp)
    else
        cities_to_try = rand(cities(tsp), ntry)
    end
    best_cost = Inf64
    best_tour = [1, 2]
    for start in cities_to_try
        tour, cost = nearestneighbors(tsp, start=start)
        if cost < best_cost
            best_cost = cost
            best_tour = tour
        end
    end
    return best_tour, best_cost
end

tour_nn, cost_nn = nearestneighbors(tsp)
tour_nnbest, cost_nnbest = bestnearestneighbors(tsp, ntry=100)  # try 100 starts

p_nn = plot_cities(tsp, markersize=1)
plot_tour!(tsp, tour_nn, label="random")
plot_tour!(tsp, tour_nnbest, label="best", color=mygreen)
title!("Nearest neighbor\n cost = $cost_nnbest")

# GREEDY

using DataStructures

"""
    greedy(tsp::TravelingSalesmanProblem)

Uses the greedy algorithm to solve the TSP.
"""
function greedy(tsp::TravelingSalesmanProblem)
    usf = DisjointSets(cities(tsp))
    times_added = Dict(c=>0 for c in cities(tsp))
    n = length(tsp)
    edges = [(dist(tsp, i, j), i, j) for i in 1:n-1 for j in (i+1):n]
    cost = 0.0
    selected_edges = Tuple{Int,Int}[]
    for (c, i, j) in sort!(edges)
        if !in_same_set(usf, i, j) && times_added[i] < 2 && times_added[j] < 2
            union!(usf, i, j)
            push!(selected_edges, (i, j))
            times_added[i] += 1
            times_added[j] += 1
            cost += c
            length(selected_edges) == n  && break
        end
    end
    # close the loop
    current = findfirst(c->times_added[c]==1, cities(tsp))
    # knit edges in a tour
    tour = [current]
    sizehint!(tour, n)
    used = Set([current])
    while length(tour) < n
        for (i, j) in selected_edges
            if i==current && j ∉ used
                push!(tour, j)
                push!(used, j)
                current = j
                break
            elseif j==current && i ∉ used
                push!(tour, i)
                push!(used, i)
                current = i
                break
            end
        end
    end
    return tour, cost
end

tour_greedy, cost_greedy = greedy(tsp)

p_greedy = plot_cities(tsp, markersize=1)
plot_tour!(tsp, tour_greedy)
title!("Greedy\n cost = $cost_greedy")

"""
    insertion(tsp::TravelingSalesmanProblem)

Randomly inserts cities in a tour at a place where it has the lowest cost.
"""
function insertion(tsp::TravelingSalesmanProblem)
    n = length(tsp)
    start = rand(cities(tsp))
    tour = [start]
    sizehint!(tour, n)
    cities_to_try = Set(cities(tsp))
    delete!(cities_to_try, start)
    cost = 0.0
    while length(tour) < n
        city = rand(cities_to_try)
        connection_cost = Inf64
        best_pos = 0
        # find cheapest place to insert city
        for pos in 1:length(tour)
            ci = (pos > 1) ? tour[pos-1] : tour[end]
            cj = tour[pos]
            # cost of connecting
            Δc = dist(tsp, ci, city) + dist(tsp, city, cj) - dist(tsp, ci, cj)
            if Δc < connection_cost
                connection_cost = Δc
                best_pos = pos
            end
        end
        insert!(tour, best_pos, city)
        delete!(cities_to_try, city)
        cost += connection_cost
    end
    return tour, computecost(tsp, tour)
end


tour_insertion, cost_insertion = insertion(tsp)

p_insertion = plot_cities(tsp, markersize=1)
plot_tour!(tsp, tour_insertion)
title!("Insertion (random)\ncost=$cost_insertion")


# 2-opt

"""
    hillclimbing!(tsp, tour; verbose=false, maxitter=Inf)

Uses hill climbing to improve a tour by finding each iteration the best path
between two cities to flip.
"""
function hillclimbing!(tsp, tour; verbose=false, maxitter=Inf)
    n = length(tsp)
    improved = true
    cost = computecost(tsp, tour)
    iter = 0
    costs = [cost]
    while improved
        best_Δc = 0.0  # any change is sufficient to continue
        improved = false
        best_i, best_j = 0, 0
        for i in 1:n-2
            for j in (i+1):n
                Δc = deltaflipcost(tsp, tour, i, j)
                if Δc < best_Δc
                    best_Δc = Δc
                    best_i, best_j = i, j
                    improved = true
                end
            end
        end
        !improved && break
        flip!(tour, best_i, best_j)
        iter += 1
        cost += best_Δc
        push!(costs, cost)
        verbose && println("Iteration $iter: Δcost = $best_Δc (flipped $best_i, $best_j)")
        iter > maxitter && break
    end
    println("converged in $iter steps")
    return tour, cost, costs
end

"""
    hillclimbing(tsp; verbose=false, maxitter=Inf)

Uses hill climbing to improve a tour by finding each iteration the best path
between two cities to flip. Starts from the given order of the cities.
"""
hillclimbing(tsp; kwargs...) = hillclimbing!(tsp, collect(1:length(tsp)); kwargs...)

tour_hc, cost_hc, costs_hc = hillclimbing(tsp, verbose=true)

p_hc = plot_cities(tsp, markersize=1)
plot_tour!(tsp, tour_hc)
title!("Hill climbing\ncost=$cost_hc")

# SIMULATED ANNEALING

"""
    simulatedannealing!tsp, tour;
                    Tmax, Tmin, r, kT::Int, verbose=false)

Uses simulated annealing to improve a tour by finding each iteration the best path
between two cities to flip.
"""
function simulatedannealing!(tsp, tour;
                Tmax, Tmin, r, kT::Int, verbose=false)
    n = length(tsp)
    improved = true
    cost = computecost(tsp, tour)
    T = Tmax
    n_steps = (log(Tmin) - log(Tmax)) / log(r) |> ceil |> Int
    costs = Vector{typeof(cost)}(undef, n_steps)
    iter = 0
    while T > Tmin
        iter += 1
        acc = 0
        for k in 1:kT
            # choose two cities to swap
            i, j = rand(1:n, 2)
            # improvement
            Δc = deltaflipcost(tsp, tour, i, j)
            if Δc < 0.0 || rand() < exp(- Δc / T)
                cost += Δc
                flip!(tour, i, j)
                acc += 1
            end
        end
        verbose && println("T = $T : cost = $cost (acc % = $(acc / kT))")
        costs[iter] = cost
        T *= r
    end
    return tour, cost, costs
end


simulatedannealing(tsp; kwargs...) = simulatedannealing!(tsp,
                                            collect(1:length(tsp)); kwargs...)

tour_sa = collect(1:n)

tour_sa, cost_sa, costs_sa = simulatedannealing!(tsp, tour_sa;
                Tmax=1e7, Tmin=1e-4, r=0.95, kT=100000, verbose=true)

p_sa = plot_cities(tsp, markersize=1)
plot_tour!(tsp, tour_sa)
title!("Simulated Annealing\ncost=$cost_sa")


# TABU SEARCH

"""
    tabusearch!(tsp::TravelingSalesmanProblem, tour; ntabu::Int, niter::Int,
                            verbose=false)

Improves a tour by iteratively performing the best local improvement, similarly
to `hillclimbing`. In tabu search however, after a position of the tour is
modified, it is blocked for `ntabu` steps. This behaviour is meant to escape
local minima.
"""
function tabusearch!(tsp::TravelingSalesmanProblem, tour; ntabu::Int, niter::Int,
                            verbose=false)
    n = length(tsp)
    cost = computecost(tsp, tour)
    # positions in the tour that are tabood
    tabood = zeros(Int, n)
    costs = Vector(undef, niter)
    for iter in 1:niter
        best_Δc = Inf64
        best_i, best_j = 0, 0
        for i in 1:n-2
            # if i is tabood skip it
            tabood[i] > iter && continue
            for j in (i+1):n
                # if j is tabood skip it
                tabood[j] > iter && continue
                Δc = deltaflipcost(tsp, tour, i, j)
                if Δc < best_Δc
                    best_Δc = Δc
                    best_i, best_j = i, j
                end
            end
        end
        flip!(tour, best_i, best_j)
        cost += best_Δc
        # tabu these for ntabu steps
        tabood[best_i] = iter + ntabu
        tabood[best_j] = iter + ntabu
        costs[iter] = cost
        verbose && println("Iteration $iter: Δcost = $best_Δc (flipped $best_i, $best_j)")
    end
    return tour, cost, costs
end

tabusearch(tsp; kwargs...) = tabusearch!(tsp, collect(1:length(tsp)); kwargs...)

tour_tabu, cost_tabu, costs_tabu = tabusearch(tsp, ntabu=50,
                niter=2_000, verbose=true)

p_tabu = plot_cities(tsp, markersize=1)
plot_tour!(tsp, tour_tabu)
title!("Tabu search\ncost=$cost_tabu")

# SUMMARIZE
# ---------

plot(p_nn, p_greedy, p_insertion, p_hc, p_sa, p_tabu, size=(1000, 1000))
savefig("figures/tsp_algorithms_$name.png")


plot(costs_hc, label="hill climbing")
plot!(costs_sa, label="simulated annealing")
plot!(costs_tabu, label="tabu search")
xlabel!("Iteration")
ylabel!("Tour cost")
savefig("figures/tsp_convergence_$name.png")
