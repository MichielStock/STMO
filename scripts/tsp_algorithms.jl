#=
Created on Sunday 19 April 2020
Last update: Tuesday 21 April 2020

@author: Michiel Stock
michielfmstock@gmail.com

Illustration of the algorithms for solving TSP.

Currently implements:
    - nearest neighbors
    - greedy
    - random insertion
=#

using STMO
using STMO.TSP

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
    sizehint!(tour, n)
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
            δ = dist(tsp, ci, city) + dist(tsp, city, cj) - dist(tsp, ci, cj)
            if δ < connection_cost
                connection_cost = δ
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

function swap(tsp, tour, )
