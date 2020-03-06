#=
Created on Tuesday 04 February 2020
Last update: Wedneday 05 February 2020

@author: Michiel Stock
michielfmstock@gmail.com

Dijkstra, A* and Bellman FOrd
=#

module ShortestPath

export reconstruct_path, dijkstra, a_star

using DataStructures, STMO

"""
    reconstruct_path(previous::Dict{T,T}, source::T, sink::T) where {T}

Reconstruct the path from the output of the Dijkstra algorithm.

Inputs:
        - previous : a Dict with the previous node in the path
        - source : the source node
        - sink : the sink node
Ouput:
        - the shortest path from source to sink
"""
function reconstruct_path(previous::Dict{T,T}, source::T, sink::T) where {T}
    path = T[sink]
    current = sink
    while current != source
        current = previous[current]
        push!(path, current)
    end
    return reverse!(path)
end


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

"""
    dijkstra(graph::AdjList{R,T}, source::T, sink::T) where {R<:Real,T}

Dijkstra's shortest path algorithm.

Inputs:
    - `graph` : adjacency list representing a weighted directed graph
    - `source`
    - `sink`

Outputs:
    - the shortest path
    - the cost of this shortest path
"""
function dijkstra(graph::AdjList{R,T}, source::T, sink::T) where {R<:Real,T}
    # initialize the tentative distances
    distances = Dict(v => Inf for v in keys(graph))
    distances[source] = 0.0
    previous = Dict{T,T}()
    vertices_to_check = [(0.0, source)]
    while length(vertices_to_check) > 0
        dist, u = heappop!(vertices_to_check)
        if u == sink
            return reconstruct_path(previous, source, sink), dist
        end
        for (dist_u_v, v) in graph[u]
            new_dist = dist + dist_u_v
            if distances[v] > new_dist
                distances[v] = new_dist
                previous[v] = u
                heappush!(vertices_to_check, (new_dist, v))
            end
        end
    end
end

"""
    a_star(graph::AdjList{R,T}, source::T, sink::T, heuristic) where {R<:Real,T}

A* shortest path algorithm.

Inputs:
    - `graph` : adjacency list representing a weighted directed graph
    - `source`
    - `sink`
    - `heuristic` : a function that inputs a node and returns an lower bound
            for the distance to the source. Note that a distance can be turned into
            a heuristic using `n -> d(n, sink)`

Outputs:
    - the shortest path
    - the cost of this shortest path
"""
function a_star(graph::AdjList{R,T}, source::T, sink::T, heuristic) where {R<:Real,T}
    # initialize the tentative distances
    distances = Dict(v => Inf for v in keys(graph))
    distances[source] = 0.0
    previous = Dict{T,T}()
    vertices_to_check = [(heuristic(source), source)]
    while length(vertices_to_check) > 0
        dist, u = heappop!(vertices_to_check)
        if u == sink
            return reconstruct_path(previous, source, sink), distances[sink]
        end
        for (dist_u_v, v) in graph[u]
            new_dist = dist + dist_u_v
            if distances[v] > new_dist
                distances[v] = new_dist
                min_dist_to_sink = new_dist + heuristic(v)
                previous[v] = u
                heappush!(vertices_to_check, (min_dist_to_sink, v))
            end
        end
    end
end


end  # module ShortestPaths
