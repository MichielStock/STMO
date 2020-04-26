#=
Created on Sunday 19 April 2020
Last update: Sunday 16 April 2020

@author: Michiel Stock
michielfmstock@gmail.com

Utilities for the TSP problem.
=#

module TSP

import STMO:dist
using STMO
using Plots: plot!, scatter, scatter!, plot

export TravelingSalesmanProblem, cities, computecost, coordinates
export plot_cities, plot_cities!, plot_tour, plot_tour!
export swap!, deltaswapcost, deltaflipcost, flip!
export totoro_tsp, got_coords

struct TravelingSalesmanProblem{Tc}
    coordinates::Matrix{Tc}
    distance::Matrix{Float64}
end

TravelingSalesmanProblem(coordinates) = TravelingSalesmanProblem(coordinates,
                                dist(coordinates, coordinates))

"""Returns the number of cities."""
Base.length(tsp::TravelingSalesmanProblem) = size(tsp.coordinates, 1)

"""Returns the coordinates of the cities."""
coordinates(tsp::TravelingSalesmanProblem) = tsp.coordinates

"""Returns the cities"""
cities(tsp::TravelingSalesmanProblem) = collect(1:length(tsp))

"""Returns the distance matrix of the cities."""
dist(tsp::TravelingSalesmanProblem) = tsp.distance

"""Returns the distance (or cost) of going from city `ci` to city `cj`"""
dist(tsp::TravelingSalesmanProblem, ci, cj) = dist(tsp)[ci,cj]

Base.isvalid(tsp::TravelingSalesmanProblem, tour) = length(tour) == length(tsp) &&
    Set(tour) == Set(cities(tsp))

"""
    computecost(tsp::TravelingSalesmanProblem, tour)

Computes the cost of travessing a tour.
"""
function computecost(tsp::TravelingSalesmanProblem, tour)
    !isvalid(tsp, tour) && throw(AssertionError("invalid tour provided"))
    c = 0.0
    for (i, j) in zip(tour[1:end-1], tour[2:end])
        c += dist(tsp, i, j)
    end
    # complete tour
    c += dist(tsp, tour[end], tour[1])
    return c
end

split_coord(X) = X[:,1], X[:,2]

plot_cities(tsp::TravelingSalesmanProblem; kwargs...) = scatter(split_coord(coordinates(tsp))...,
                color=myblue, label="", aspect_ratio=:equal; kwargs...)

plot_cities!(tsp::TravelingSalesmanProblem; kwargs...) = scatter!(split_coord(coordinates(tsp))...,
                color=myblue, label=""; kwargs...)

coords_tour(tsp, tour) = [coordinates(tsp)[tour,:];coordinates(tsp)[[tour[1]],:]]

plot_tour(tsp::TravelingSalesmanProblem, tour; kwargs...) = plot(
                split_coord(coords_tour(tsp, tour))...,
                color=myred, label="", aspect_ratio=:equal; kwargs...)

plot_tour!(tsp::TravelingSalesmanProblem, tour; kwargs...) = plot!(
                split_coord(coords_tour(tsp, tour))...,
                color=myred, label=""; kwargs...)


"""
    deltaswapcost(tsp, tour, i, j)

Compute the change in tour cost if the cities at positions `i` and `j` are
swapped.
"""
function deltaswapcost(tsp, tour, i, j)
    n = length(tsp)
    i == j && return 0.0
    # put in order
    i, j = i < j ? (i, j) : (j, i)
    # choose indices respecting cyclic  invariance
    i₋₁ = i == 1 ? n : i - 1
    i₊₁ = i == n ? 1 : i + 1
    j₋₁ = j == 1 ? n : j - 1
    j₊₁ = j == n ? 1 : j + 1
    ci₋₁, ci, ci₊₁, cj₋₁, cj, cj₊₁ = tour[[i₋₁, i, i₊₁, j₋₁, j, j₊₁]]
    if j - i == 1  # i and j are neighbors
        Δc = ((dist(tsp, ci₋₁, cj) + dist(tsp, ci, cj₊₁)) -
                    (dist(tsp, ci₋₁, ci) + dist(tsp, cj, cj₊₁)))
    elseif (i==1 && j==n)
        Δc = ((dist(tsp, tour[end-1], tour[1]) + dist(tsp, tour[end], tour[2])) -
                    (dist(tsp, tour[end-1], tour[end]) + dist(tsp, tour[1], tour[2])))
    else
        Δc = ((dist(tsp, ci₋₁, cj) + dist(tsp, cj, ci₊₁) +
                    dist(tsp, cj₋₁, ci) + dist(tsp, ci, cj₊₁)) -
                    (dist(tsp, ci₋₁, ci) + dist(tsp, ci, ci₊₁) +
                    dist(tsp, cj₋₁, cj) + dist(tsp, cj, cj₊₁)))
    end
    return Δc
end

"""
    swap!(tour, i, j)

Swaps the cities at positions `i` and `j` in `tour`.
"""
function swap!(tour, i, j)
    tour[i], tour[j] = tour[j], tour[i]
    return tour
end

"""
    deltaflipcost(tsp, tour, i, j)

Compute the change in tour cost if the subtour between the `i`th and `j`th city
are flipped.
"""
function deltaflipcost(tsp, tour, i, j)
    i == j && return 0.0
    n = length(tsp)
    # put in order
    i, j = i < j ? (i, j) : (j, i)
    (i, j) == (1, n) && return 0.0
    # choose indices respecting cyclic  invariance
    i₋₁ = i == 1 ? n : i - 1
    i₊₁ = i == n ? 1 : i + 1
    j₋₁ = j == 1 ? n : j - 1
    j₊₁ = j == n ? 1 : j + 1
    ci₋₁, ci, ci₊₁, cj₋₁, cj, cj₊₁ = tour[[i₋₁, i, i₊₁, j₋₁, j, j₊₁]]
    Δc = - (dist(tsp, ci₋₁, ci) + dist(tsp, cj, cj₊₁))
    Δc += dist(tsp, ci₋₁, cj) + dist(tsp, ci, cj₊₁)
    return Δc
end

function flip!(tour, i, j)
    i, j = i < j ? (i, j) : (j, i)
    reverse!(@view tour[i:j])
    return tour
end

# example network
include("targ.jl")
include("totoro.jl")

got_tsp() = TravelingSalesmanProblem(got_coords)
totoro_tsp() = TravelingSalesmanProblem(totoro_coords)

end  # module TSP
