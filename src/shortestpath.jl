#=
Created on Tuesday 04 February 2020
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Dijkstra, A* and Bellman FOrd
=#

module ShortestPath

export dijkstra

using DataStructures, STMO

"""
    dijkstra(graph::AdjList{R,T}, source::T) where {R<:Real,T}

Dijkstra without a specified sink. Give the `graph` and a `source` and this
function uses Dijkstra's algorithm to compute all distances between the nodes
and the source. Returns
    - `distances`: a dictionary with the distances
    - `previous`: a dictionary representing the tree of the shortest paths.
"""
function dijkstra(graph::AdjList{R,T}, source::T) where {R<:Real,T}
    # initialize the tentative distances
    distances = Dict(v => Inf for v in keys(graph))
    distances[source] = 0.0
    previous = Dict{T,T}()
    vertices_to_check = [(0.0, source)]
    while length(vertices_to_check) > 0
        dist, u = heappop!(vertices_to_check)
        for (dist_u_v, v) in graph[u]
            new_dist = dist + dist_u_v
            if distances[v] > new_dist
                distances[v] = new_dist
                previous[v] = u
                heappush!(vertices_to_check, (new_dist, v))
            end
        end
    end
    return distances, previous
end

end  # module ShortestPaths
