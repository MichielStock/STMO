#=
Created on Sunday 19 April 2020
Last update: -

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
export totoro_tsp, got_coords

struct TravelingSalesmanProblem{Tc,Td}
    coordinates::Matrix{Tc}
    distance::Matrix{Td}
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

"""Returns the distance (or cost) of going from city `i` to city `j`"""
dist(tsp::TravelingSalesmanProblem, i, j) = dist(tsp)[i,j]

Base.isvalid(tsp::TravelingSalesmanProblem, tour) = length(tour) == length(tsp) &&
    Set(tour) == Set(cities(tsp))

"""
    computecost(tsp::TravelingSalesmanProblem, tour)

Computes the cost of travessing a tour.
"""
function computecost(tsp::TravelingSalesmanProblem{Tc,Td}, tour) where {Tc,Td}
    !isvalid(tsp, tour) && throw(AssertionError("invalid tour provided"))
    c = zero(Td)
    for (i, j) in zip(tour[1:end-1], tour[2:end])
        c += dist(tsp, i, j)
    end
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

# example network
include("targ.jl")
include("totoro.jl")

got_tsp() = TravelingSalesmanProblem(got_coords)
totoro_tsp() = TravelingSalesmanProblem(totoro_coords)

end  # module TSP
