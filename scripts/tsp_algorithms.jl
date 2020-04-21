#=
Created on Sunday 19 April 2020
Last update:

@author: Michiel Stock
michielfmstock@gmail.com

Illustration of the algorithms for solving TSP.

Currently implements:
    - nearest neighbors
    - greedy
=#

using STMO.TSP
using STMO
using Plots

tsp = totoro_tsp()
name = "totoro"

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
    for start in cities(tsp)
        tour, cost = nearestneighbors(tsp, start=start)
        if cost < best_cost
            best_cost = cost
            best_tour = tour
        end
    end
    return best_tour, best_cost
end

tour_nn, cost_nn = nearestneighbors(tsp)
tour_nnbest, cost_nnbest = bestnearestneighbors(tsp)

p_nn = plot_cities(tsp, markersize=2)
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

p_greedy = plot_cities(tsp, markersize=2)
plot_tour!(tsp, tour_greedy)
title!("Greedy\n cost = $cost_greedy")


function insertion_farthest(tsp::TravelingSalesmanProblem;
                start=nothing)
    if start isa Nothing
        current = rand(cities(tsp))
    else
        current = start
    end
    n = length(tsp)
    tour = [current]
    cities_to_try = Set(cities(tsp))
    delete!(cities_to_try, current)
    while length(tour) < n
        best_crit = -Inf
        best_city = 0
        best_pos = 0
        for c in cities_to_try
            for pos in 1:length(tour)-1
                crit = criterion(tsp, tour, pos, c)
                if crit > best_crit
                    best_crit = crit
                    best_pos = pos
                    best_city = c
                end
            end
        end
        insert!(tour, best_pos, best_city)
        delete!(cities_to_try, best_city)
    end
    return tour, computecost(tsp, tour)
end
