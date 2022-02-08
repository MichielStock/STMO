### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ be17c905-4bc7-4e1a-8132-3c0626fd4e08
using Plots, DataStructures

# ╔═╡ bf1571b3-9840-4bb0-a623-ccc53b49c177
module Solution

using DataStructures

export prim, kruskal

# special types for graphs
EdgeList{T} = Array{Tuple{T,T},1}
WeightedEdgeList{R,T} = Array{Tuple{R,T,T},1}
Vertices{T} = Array{T,1}
AdjList{R,T} = Dict{T,Array{Tuple{R,T},1}}

"""
Turns a list of weighted edges in an adjacency matrix (implemented as a Dict).
If the keyword `double` is set to `true`, every edge is added twice: `(w, u, v)`
and `(w, v, u)`. This is the default behaviour.
"""
function edges2adjlist(edges::WeightedEdgeList{R,T}; double=true) where {R<:Real,T}
    adjlist = AdjList{R,T}()
    for (w, i, j) in edges
        if !haskey(adjlist, i); adjlist[i] = [] end
        if !haskey(adjlist, j); adjlist[j] = [] end
        push!(adjlist[i], (w, j))
        double && push!(adjlist[j], (w, i))
    end
    return adjlist
end

"""
Turns an adjacency list (implemented as a Dict) into an edge list.
"""
adjlist2edges(adjlist::AdjList{R,T}) where {R<:Real, T} =
            [(w, v, n) for (v, neighbors) in adjlist for (w, n) in neighbors]

"""
Returns the number of vertices in a graph.
"""
nvertices(adjlist::AdjList) = length(adjlist)

"""
Returns the vertices of a graph.
"""
vertices(adjlist::AdjList) = Set(keys(adjlist))

"""
Returns the vertices of a graph.
"""
function vertices(edgelist::WeightedEdgeList{R,T}) where {R<:Real,T}
    vertices = Set{T}()
    for (w, u, v) in edgelist
        push!(vertices, u)
        push!(vertices, v)
    end
    return vertices
end

"""
Returns the number of vertices in a graph.
"""
nvertices(edgelist::WeightedEdgeList) = length(vertices(edgelist))

function isconnected(adjlist::AdjList{R,T}) where {R<:Real, T}
    visited = Set{T}()
    to_explore = [first(keys(adjlist))]
    while length(to_explore) > 0
        u = pop!(to_explore)
        push!(visited, u)
        for (w, n) in adjlist[u]
            n ∉ visited && push!(to_explore, n)
        end
    end
    return length(visited) == nvertices(adjlist)
end

isconnected(edges::WeightedEdgeList) = isconnected(edges2adjlist(edges))

"""
    prim(vertices, edges[, start])

Prim's algorithm for finding the minimum spanning tree. Inputs the vertices
(`vertices`), a list of weighted edges (`vertices`) and a starting vertex (`start`).
A random starting vertex will be chosen by default.
"""
function prim(vertices, edges,
                start=rand(vertices))
    u = start
    adjlist = edges2adjlist(edges)
    mst_edges = eltype(edges)[]
    mst_vertices = Set([u])
    edges_to_check = [(w, u, n) for (w, n) in adjlist[u]]
    cost = zero(edges[1][1])
    heapify!(edges_to_check)
    while length(mst_edges) < length(vertices) && length(edges_to_check) > 0
        # pop shortest edge, u part of tree, v might be new
        w, u, v = heappop!(edges_to_check)
        if v ∉ mst_vertices
            # add to MST
            push!(mst_edges, (w, u, v))
            push!(mst_vertices, v)
            # update cost
            cost += w
            # add neighbours of v
            for (wn, n) in adjlist[v]
                n ∉ mst_vertices && heappush!(edges_to_check, (wn, v, n))
            end
        end
    end
    return mst_edges, cost
end



"""
    kruskal(vertices, edges)

Kruskal's algorithm for finding the minimum spanning tree. Inputs the vertices
(`vertices`) and a list of weighted edges (`vertices`).
"""
function kruskal(vertices, edges)
    usf = DisjointSets(vertices)
    mst_edges = eltype(edges)[]
    mst_vertices = Set{eltype(vertices)}()
	# sort the edges by weight
    sort!(edges)
    cost = zero(edges[1][1])
	# for all edges
    for (w, u, v) in edges
		# only connect subtrees not yet connected
		# otherwise, you generate a cycle
        if !in_same_set(usf, u, v)
            push!(mst_edges, (w, u, v))
            union!(usf, u, v)
            push!(mst_vertices, u)
            push!(mst_vertices, v)
            cost += w
            if length(mst_vertices) == length(vertices)
                break
            end
        end
    end
    return mst_edges, cost
end



end  # module MST


# ╔═╡ cc4ba764-1c81-11ec-25ba-bbe4fc081aaa
md"""
# Minimal spanning trees

*STMO*

**Michiel Stock**

![](https://github.com/MichielStock/STMO/blob/master/chapters/07.MST/Figures/logo.png?raw=true)
"""

# ╔═╡ 2eb01866-ce3a-4866-bfcf-155489efa173
md"In this chapter, we will introduce minimization problems on graphs. We will study an elementary problem in computer science: finding the *minimum spanning tree* (MST). The minimum spanning tree has plenty of real-world applications, such as designing computer-, telecommunication- or other supply networks, computer graphics, and bioinformatics. Interestingly, there are efficient algorithms that can find the minimum spanning tree for even huge problems! Along our way, we will also be getting acquainted with some new types of data structures other than simple matrices.
"

# ╔═╡ e6cc2cc2-af23-4676-8422-cc3ddd964034
md"
## Graphs in Julia

Graphs are principal tools in computer science. Most programming languages provide interfaces for graphs; Julia is no exception. A simple package for working with is [`LightGraphs.jl`](https://github.com/JuliaGraphs/LightGraphs.jl). In this course, we will limit ourselves to using the basic data structures provided by these languages, arrays, sets, and dictionaries. The type system allows us to formally encode how we will represent graph structures.
"

# ╔═╡ 0cb7f29e-17f8-4234-9c08-7b17a7cd5acf
EdgeList{T} = Array{Tuple{T,T},1}

# ╔═╡ 34e3960e-52f3-48e0-b8a8-4b739aa1005f
WeightedEdgeList{R,T} = Array{Tuple{R,T,T},1}

# ╔═╡ 447d22ff-89b1-4b1c-ad7c-6be373367108
Vertices{T} = Array{T,1}

# ╔═╡ c99e1156-1f0d-479c-bd54-9dd3aaa936b3
AdjList{R,T} = Dict{T,Array{Tuple{R,T},1}}

# ╔═╡ 3fef1399-0d42-4785-91cd-8db2b67fbee6
md"""
Don't worry if you don't fully understand the above. It makes use of the type system and it allows us to use dispatch to select the best function.

Consider the following example graph:

![A small example to show how to implement graphs in Julia.](https://github.com/MichielStock/STMO/blob/master/chapters/07.MST/Figures/graph.png?raw=true)

This graph can be represented using an *adjacency list*. We do this using a `Dict`. Every vertex is a key with the adjacent vertices given as a `set` containing tuples `(weight, neighbor)`. The weight is first because this makes it easy to compare the weights of two edges. Note that for every ingoing edge, there is also an outgoing edge, this is an undirected graph.
"""

# ╔═╡ 63ebac28-66a7-4e7c-8576-1e96bb0df0fa
graph = Dict(
    'A' => [(2, 'B'), (3, 'D')],
    'B' => [(2, 'A'), (1, 'C'), (2, 'E')],
    'C' => [(1, 'B'), (2, 'D'), (1, 'E')],
    'D' => [(2, 'C'), (3, 'A'), (3, 'E')],
    'E' => [(2, 'B'), (1, 'C'), (3, 'D')]
)

# ╔═╡ 8c25749b-8fec-4404-bc1b-b2e15edd4d58
graph isa AdjList

# ╔═╡ 675e5833-081e-429d-9ae6-adc51c62e2bf
md"Sometimes we will use an *edge list*, i.e., a list of (weighted) edges. The edge list is often a more compact way of storing a graph. We show the edge list for the example graph below. Note that again every edge is double: we include an in- and outgoing edge."

# ╔═╡ 1f8aa16e-9a2a-41de-a6e6-3a5b37121e18
edges = [
 (2, 'B', 'A'),
 (3, 'D', 'A'),
 (2, 'C', 'D'),
 (3, 'A', 'D'),
 (3, 'E', 'D'),
 (2, 'B', 'E'),
 (3, 'D', 'E'),
 (1, 'C', 'E'),
 (2, 'E', 'B'),
 (2, 'A', 'B'),
 (1, 'C', 'B'),
 (1, 'E', 'C'),
 (1, 'B', 'C'),
 (2, 'D', 'C')];

# ╔═╡ d0be86e9-0cdb-4dbd-b6dd-536c921ea56d
md"""
## Some useful data structures

### Disjoint-set data structure

Implementing an algorithm for finding the minimum spanning tree is reasonably straightforward. The only bottleneck is that the algorithm requires the disjoint-set data structure to keep track of a set partitioned in several disjoined subsets.

For example, consider the following initial set of eight elements.

![](https://github.com/MichielStock/STMO/blob/master/chapters/07.MST/Figures/disjointset1.png?raw=true)

We decide to group elements A, B, and C together in a subset and F and G in another subset.

![](https://github.com/MichielStock/STMO/blob/master/chapters/07.MST/Figures/disjointset2.png?raw=true)

The disjoint-set data structure supports the following operations:

- **Find**: check which subset an element is in. Is typically used to check whether two objects are in the same subset;
- **Union** merges two subsets into a single subset.

A Julia implementation of a disjoint-set is available in the `DataStructures` library. The function `DisjointSets` turns a list in a union set forest. The function `union!` will merge the sets of two elements while `in_same_set` can be used to check whether two items are in the same set. A simple example will make everything clear!

"""

# ╔═╡ 27c15a60-e6b2-4cf4-86e9-ee7bbfce4abc
animals = ["mouse", "bat", "robin", "trout", "seagull", "hummingbird",
           "salmon", "goldfish", "hippopotamus", "whale", "sparrow"]

# ╔═╡ 906c4cef-639c-46b0-950e-f0b113ca6bed
union_set_forest = DisjointSets(animals)

# ╔═╡ 2f70b1f4-b494-4d91-ade9-3164ee7b3e4a
md"Let us do some operations on the USF!"

# ╔═╡ 4984d6c4-94c2-470a-90f9-90873e2e65df
begin
	# group mammals together
	union!(union_set_forest, "mouse", "bat")
	union!(union_set_forest, "mouse", "hippopotamus")
	union!(union_set_forest, "whale", "bat")
	
	# group birds together
	union!(union_set_forest, "robin", "seagull")
	union!(union_set_forest, "seagull", "sparrow")
	union!(union_set_forest, "seagull", "hummingbird")
	union!(union_set_forest, "robin", "hummingbird")
	
	# group fishes together
	union!(union_set_forest, "goldfish", "salmon")
	union!(union_set_forest, "trout", "salmon")
end

# ╔═╡ 21dae9d6-1ef9-4213-b7ed-6a4f98336c16
# mouse and whale in same subset?
in_same_set(union_set_forest, "mouse", "whale")

# ╔═╡ a61124a5-33e8-4b8f-bdf0-53417adb99ca
# mouse and whale in same subset?
in_same_set(union_set_forest, "robin", "salmon")

# ╔═╡ 7d451006-6199-478b-83d6-e3432d5eee6d
md"""
### Heap queue

One can use a heap queue to find the minimum of a changing list without having to sort the list every update. Heaps are also implemented in `DataStructures`. The function `heapify!` will rearrange a list to satisisfy the heap property. `heappop!` and `heappush!` can be used to extract, resp. add, elements while maintaining the heap property.
"""

# ╔═╡ 32fb96a6-9a1f-4e53-b618-a63440c3231f
heap = [(5, 'A'), (3, 'B'), (2, 'C'), (7, 'D')]

# ╔═╡ feed3e33-b48a-45ca-9786-95c52a1601c8
md"Turn a list into a heap:"

# ╔═╡ 8ff87d66-1cf7-4b0d-a0e8-dc2eb15c3a16
heapify!(heap)

# ╔═╡ e270de77-8537-49ae-b1bf-c986d054afe7
md"Return item lowest value while retaining heap property:"

# ╔═╡ 5658dee9-65e0-4fb8-9584-a5090d207bed
heappop!(heap)

# ╔═╡ c668a422-3aa1-4061-9f05-af4f09188470
heap

# ╔═╡ cffe5be7-bf2e-4d97-8ac6-2d0ce30bab9a
md"Add new item and retain heap property:"

# ╔═╡ 13f2f4d8-6afb-454f-b8d9-fb15dbed0200
heappush!(heap, (4, 'E'))

# ╔═╡ 9645b70d-06ce-4b2c-a980-2ba9e3068799
md"""
## Two algorithms for finding minimum spanning trees

### Prim's algorithm

Prim's algorithm starts with a single vertex and adds $|V|-1$ edges to it, always taking the next edge with a minimal weight that connects a vertex on the MST to a vertex not yet in the MST. Complete the code below.
"""

# ╔═╡ f7adca17-925c-4ee5-ad93-8d2b27c3157c
md"""
### Kruskal's algorithm


Kruskal's algorithm is a straightforward algorithm to find the minimum spanning tree. The main idea is to start with an initial 'forest' of the individual nodes of the graph. In each step of the algorithm, we add an edge with the smallest possible value that connects two disjoint trees in the forest. This process is continued until we have a single tree, which is a minimum spanning tree, or until all edges are considered. In the latter case, the algorithm returns a minimum spanning forest.
"""

# ╔═╡ 1f49a5fa-8232-4901-88bb-9ffa382e5fc1
"""
    kruskal(vertices, edges)

Kruskal's algorithm for finding the minimum spanning tree. Inputs the vertices
(`vertices`) and a list of weighted edges (`vertices`).
"""
function kruskal(vertices, edges)
    missing  # complete this
    return mst_edges, cost
end

# ╔═╡ a44b2f20-98a6-4162-b48c-2725b3663e69
md"""
## Ticket to ride

As an illustration, we provide the graph of the famous boardgame *Ticket To Ride* (USA version). The goal of this game is to connect two cities on a map by placing a number of trains between them. Let's load the graph!
"""

# ╔═╡ d2c92e1d-a3cb-4070-8f2c-cf01e943ad45
md"The weighted edges. The weight represents the connection cost."

# ╔═╡ cda14c79-7a39-4ef5-8c59-1dd2f0fac202
md"Let us plot this graph. We also have the coordinates of the cities in `cities_coordinates`. It is not needed to find the MST, but can help us make a draw a map of the USA."

# ╔═╡ 44609196-e3f2-446b-9ca9-d8741c43fb05
md"Your turn! Use the functions above to find a minimal spanning tree for this graph."

# ╔═╡ f50f0928-7598-4e36-9a65-876a42b8982e


# ╔═╡ 2febee75-48eb-4226-acaf-f33b36eb19bc


# ╔═╡ d16daa50-aff5-447c-8aaf-18da9b57301d
md"## Appendix"

# ╔═╡ 454fcd33-52c0-48d8-ba15-ffb5b0249ac4
begin
	myblue = "#304da5"
	mygreen = "#2a9d8f"
	myyellow = "#e9c46a"
	myorange = "#f4a261"
	myred = "#e76f51"
	myblack = "#50514F"

	mycolors = [myblue, myred, mygreen, myorange, myyellow]
end;

# ╔═╡ bbc8bc84-98db-4b0d-b50a-182ed6208027
md"### Some helpful graph functions"

# ╔═╡ 48fbc53f-4a01-4996-9761-063b5a5c8aa5
"""
Turns a list of weighted edges in an adjacency matrix (implemented as a Dict).
If the keyword `double` is set to `true`, every edge is added twice: `(w, u, v)`
and `(w, v, u)`. This is the default behaviour.
"""
function edges2adjlist(edges::WeightedEdgeList{R,T}; double=true) where {R<:Real,T}
    adjlist = AdjList{R,T}()
    for (w, i, j) in edges
        if !haskey(adjlist, i); adjlist[i] = [] end
        if !haskey(adjlist, j); adjlist[j] = [] end
        push!(adjlist[i], (w, j))
        double && push!(adjlist[j], (w, i))
    end
    return adjlist
end

# ╔═╡ 2e058829-0102-494c-a6b2-2ce708d404f4
"""
    prim(vertices, edges, start)

Prim's algorithm for finding the minimum spanning tree. Inputs the vertices
(`vertices`), a list of weighted edges (`vertices`), and a starting vertex (`start`).
By default, as random starting vertex will be chosen.
"""
function prim(vertices, edges, start=rand(vertices))
    u = start
    adjlist = edges2adjlist(edges)
    missing  # complete this
    return mst_edges, cost
end

# ╔═╡ c863a3d0-e09b-4cc8-8b32-1b66a7ca78ec
"""
Turns an adjacency list (implemented as a Dict) into an edge list.
"""
adjlist2edges(adjlist::AdjList{R,T}) where {R<:Real, T} =
            [(w, v, n) for (v, neighbors) in adjlist for (w, n) in neighbors]

# ╔═╡ ca9c8c66-99bb-406b-b41b-e237f69642fb
adjlist2edges(graph)

# ╔═╡ e294887b-e9d6-41fb-91e7-61e564c18745
"""
Returns the number of vertices in a graph.
"""
nvertices(adjlist::AdjList) = length(adjlist)

# ╔═╡ 5f52d1af-6866-4c51-9eb2-a8eb7e0043d9
"""
Returns the vertices of a graph.
"""
vertices(adjlist::AdjList) = Set(keys(adjlist))

# ╔═╡ 1b0b424e-1abe-4391-bc36-7d262ff9eaf9
"""
Returns the vertices of a graph.
"""
function vertices(edgelist::WeightedEdgeList{R,T}) where {R<:Real,T}
    vertices = Set{T}()
    for (w, u, v) in edgelist
        push!(vertices, u)
        push!(vertices, v)
    end
    return vertices
end

# ╔═╡ 9d9b793c-71db-4273-b452-89f5ff937581
"""
Returns the number of vertices in a graph.
"""
nvertices(edgelist::WeightedEdgeList) = length(vertices(edgelist))

# ╔═╡ 5f70891f-c990-44f1-9092-fc134d2903b5
function isconnected(adjlist::AdjList{R,T}) where {R<:Real, T}
    visited = Set{T}()
    to_explore = [first(keys(adjlist))]
    while length(to_explore) > 0
        u = pop!(to_explore)
        push!(visited, u)
        for (w, n) in adjlist[u]
            n ∉ visited && push!(to_explore, n)
        end
    end
    return length(visited) == nvertices(adjlist)
end

# ╔═╡ ccba1495-d81d-4414-8a0a-dcd027ec34fd
isconnected(edges::WeightedEdgeList) = isconnected(edges2adjlist(edges))

# ╔═╡ d5cca937-ffbc-4d78-853b-1379b92b91c2
md"### Data"

# ╔═╡ 4013f2d2-7324-46c8-a186-46d2855237ee
const tickettoride_edges = [(1, "Vancouver", "Seattle"),
          (1, "Seattle", "Portland"),
          (3, "Vancouver", "Calgary"),
          (6, "Calgary", "Winnipeg"),
          (6, "Winnipeg", "Sault Ste. Marie"),
          (4, "Winnipeg", "Helena"),
          (4, "Calgary", "Helena"),
          (6, "Seattle", "Helena"),
          (4, "Seattle", "Calgary"),
          (6, "Portland", "Salt Lake City"),
          (5, "Portland", "San Francisco"),
          (5, "San Francisco", "Salt Lake City"),
          (3, "San Francisco", "Los Angeles"),
          (2, "Los Angeles", "Las Vegas"),
          (3, "Los Angeles", "Phoenix"),
          (6, "Los Angeles", "El Paso"),
          (3, "Phoenix", "El Paso"),
          (3, "Phoenix", "Santa Fe"),
          (3, "Las Vegas", "Salt Lake City"),
          (5, "Phoenix", "Denver"),
          (3, "Salt Lake City", "Denver"),
          (3, "Helena", "Salt Lake City"),
          (6, "Helena", "Duluth"),
          (4, "Winnipeg", "Duluth"),
          (4, "Helena", "Denver"),
          (5, "Helena", "Omaha"),
          (4, "Denver", "Omaha"),
          (4, "Denver", "Kansas City"),
          (2, "Denver", "Santa Fe"),
          (2, "Santa Fe", "El Paso"),
          (3, "Santa Fe", "Oklahoma City"),
          (4, "Denver", "Oklahoma City"),
          (6, "El Paso", "Houston"),
          (4, "El Paso", "Dallas"),
          (5, "El Paso", "Oklahoma City"),
          (1, "Dallas", "Houston"),
          (2, "Dallas", "Oklahoma City"),
          (2, "Kansas City", "Oklahoma City"),
          (1, "Omaha", "Kansas City"),
          (2, "Omaha", "Duluth"),
          (3, "Duluth", "Chicago"),
          (4, "Omaha", "Chicago"),
          (6, "Duluth", "Toronto"),
          (3, "Duluth", "Sault Ste. Marie"),
          (5, "Sault Ste. Marie", "Montreal"),
          (2, "Montreal", "Boston"),
          (2, "Boston", "New York"),
          (3, "Montreal", "New York"),
          (3, "Montreal", "Toronto"),
          (4, "Toronto", "Chicago"),
          (3, "Chicago", "Pittsburgh"),
          (2, "Chicago", "Saint Louis"),
          (2, "Pittsburgh", "Toronto"),
          (2, "Toronto", "Sault Ste. Marie"),
          (2, "Pittsburgh", "New York"),
          (2, "Pittsburgh", "Washington"),
          (2, "Washington", "New York"),
          (2, "Washington", "Raleigh"),
          (2, "Pittsburgh", "Raleigh"),
          (5, "Pittsburgh", "Saint Louis"),
          (2, "Kansas City", "Saint Louis"),
          (2, "Nashville", "Saint Louis"),
          (2, "Little Rock", "Saint Louis"),
          (2, "Oklahoma City", "Little Rock"),
          (2, "Little Rock", "Dallas"),
          (3, "Little Rock", "Nashville"),
          (2, "Houston", "New Orleans"),
          (3, "Little Rock", "New Orleans"),
          (4, "New Orleans", "Atlanta"),
          (1, "Atlanta", "Nashville"),
          (4, "Nashville", "Pittsburgh"),
          (2, "Atlanta", "Raleigh"),
          (3, "Nashville", "Raleigh"),
          (2, "Raleigh", "Charleston"),
          (2, "Charleston", "Atlanta"),
          (6, "New Orleans", "Miami"),
          (5, "Atlanta", "Miami"),
          (4, "Charleston", "Miami")
          ]

# ╔═╡ 13b9dc04-a1ce-4aaf-8a14-f92a7c6b5da7
tickettoride_edges

# ╔═╡ 1935b2b0-84c7-4e7c-80f1-e331776a3836
const cities_coordinates = Dict("Atlanta" => (-84.3901849, 33.7490987),
                               "Boston" => (-71.0595678, 42.3604823),
                               "Calgary" => (-114.0625892, 51.0534234),
                               "Charleston" => (-79.9402728, 32.7876012),
                               "Chicago" => (-87.6244212, 41.8755546),
                               "Dallas" => (-96.7968559, 32.7762719),
                               "Denver" => (-104.9847034, 39.7391536),
                               "Duluth" => (-92.1251218, 46.7729322),
                               "El Paso" => (-106.501349395577, 31.8111305),
                               "Helena" => (-112.036109, 46.592712),
                               "Houston" => (-95.3676974, 29.7589382),
                               "Kansas City" => (-94.5630298, 39.0844687),
                               "Las Vegas" => (-115.149225, 36.1662859),
                               "Little Rock" => (-92.2895948, 34.7464809),
                               "Los Angeles" => (-118.244476, 34.054935),
                               "Miami" => (-80.1936589, 25.7742658),
                               "Montreal" => (-73.6103642, 45.4972159),
                               "Nashville" => (-86.7743531, 36.1622296),
                               "New Orleans" => (-89.9750054503052, 30.03280175),
                               "New York" => (-73.9866136, 40.7306458),
                               "Oklahoma City" => (-97.5170536, 35.4729886),
                               "Omaha" => (-95.9378732, 41.2587317),
                               "Phoenix" => (-112.0773456, 33.4485866),
                               "Pittsburgh" => (-79.99, 40.42),
                               "Portland" => (-122.6741949, 45.5202471),
                               "Raleigh" => (-78.6390989, 35.7803977),
                               "Saint Louis" => (-90.12954315, 38.60187637),
                               "Salt Lake City" => (-111.8904308, 40.7670126),
                               "San Francisco" => (-122.49, 37.75),
                               "Santa Fe" => (-105.9377997, 35.6869996),
                               "Sault Ste. Marie" => (-84.320068, 46.52391),
                               "Seattle" => (-122.3300624, 47.6038321),
                               "Toronto" => (-79.387207, 43.653963),
                               "Vancouver" => (-123.1139529, 49.2608724),
                               "Washington" => (-77.0366456, 38.8949549),
                               "Winnipeg" => (-97.168579, 49.884017))

# ╔═╡ ad534796-d5a1-4c45-b362-68754db23d85
begin
	p = plot(xaxis="longitude", yaxis="lattitude")

	# add edges
	for (w, c1, c2) in tickettoride_edges
		x1, y1 = cities_coordinates[c1]
		x2, y2 = cities_coordinates[c2]
		plot!(p, [x1, x2], [y1, y2], color=myred, alpha=0.8,
			lw=w, label="")
	end
	# plot cities
	for (city, (x, y)) in cities_coordinates
		println("$city: $x, $y")
		scatter!(p, [x], [y], label="", color=mygreen, markersize=10, alpha=0.8
		, annotations=[(x, y, city, 8)]  # comment this for clarity
		)
	end
	p
end

# ╔═╡ c7b00672-4602-4452-a8af-a4918f4a30dc
const cities = [k for k in keys(cities_coordinates)]

# ╔═╡ 66f7ec1d-a5a8-41df-b64e-80634747b7a3
cities

# ╔═╡ 0cf0cb00-7007-42be-a5b7-f06f6c6f3992
tickettoride_dist(c1, c2) = sqrt(sum((cities_coordinates[c1] .- cities_coordinates[c2]).^2))

# ╔═╡ 3c932fb9-8667-475c-841f-2a4d4aa4c3ae
tickettoride_edges_dists = [(tickettoride_dist(u, v), u, v) for (w, u, v) in tickettoride_edges]

# ╔═╡ b2371496-9da4-4c59-8efe-828844bc91a3
const tickettoride_graph = edges2adjlist(tickettoride_edges_dists)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
DataStructures = "~0.18.10"
Plots = "~1.22.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

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

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

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

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

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
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

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
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

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
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

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
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

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

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

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
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

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
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "4c2637482176b1c2fb99af4d83cb2ff0328fc33c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.1"

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
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

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
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

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
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

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

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

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
# ╟─cc4ba764-1c81-11ec-25ba-bbe4fc081aaa
# ╠═be17c905-4bc7-4e1a-8132-3c0626fd4e08
# ╟─2eb01866-ce3a-4866-bfcf-155489efa173
# ╟─e6cc2cc2-af23-4676-8422-cc3ddd964034
# ╠═0cb7f29e-17f8-4234-9c08-7b17a7cd5acf
# ╠═34e3960e-52f3-48e0-b8a8-4b739aa1005f
# ╠═447d22ff-89b1-4b1c-ad7c-6be373367108
# ╠═c99e1156-1f0d-479c-bd54-9dd3aaa936b3
# ╟─3fef1399-0d42-4785-91cd-8db2b67fbee6
# ╠═63ebac28-66a7-4e7c-8576-1e96bb0df0fa
# ╠═8c25749b-8fec-4404-bc1b-b2e15edd4d58
# ╟─675e5833-081e-429d-9ae6-adc51c62e2bf
# ╠═1f8aa16e-9a2a-41de-a6e6-3a5b37121e18
# ╠═ca9c8c66-99bb-406b-b41b-e237f69642fb
# ╟─d0be86e9-0cdb-4dbd-b6dd-536c921ea56d
# ╠═27c15a60-e6b2-4cf4-86e9-ee7bbfce4abc
# ╠═906c4cef-639c-46b0-950e-f0b113ca6bed
# ╠═2f70b1f4-b494-4d91-ade9-3164ee7b3e4a
# ╠═4984d6c4-94c2-470a-90f9-90873e2e65df
# ╠═21dae9d6-1ef9-4213-b7ed-6a4f98336c16
# ╠═a61124a5-33e8-4b8f-bdf0-53417adb99ca
# ╟─7d451006-6199-478b-83d6-e3432d5eee6d
# ╠═32fb96a6-9a1f-4e53-b618-a63440c3231f
# ╟─feed3e33-b48a-45ca-9786-95c52a1601c8
# ╠═8ff87d66-1cf7-4b0d-a0e8-dc2eb15c3a16
# ╟─e270de77-8537-49ae-b1bf-c986d054afe7
# ╠═5658dee9-65e0-4fb8-9584-a5090d207bed
# ╠═c668a422-3aa1-4061-9f05-af4f09188470
# ╟─cffe5be7-bf2e-4d97-8ac6-2d0ce30bab9a
# ╠═13f2f4d8-6afb-454f-b8d9-fb15dbed0200
# ╟─9645b70d-06ce-4b2c-a980-2ba9e3068799
# ╟─2e058829-0102-494c-a6b2-2ce708d404f4
# ╟─f7adca17-925c-4ee5-ad93-8d2b27c3157c
# ╠═1f49a5fa-8232-4901-88bb-9ffa382e5fc1
# ╟─a44b2f20-98a6-4162-b48c-2725b3663e69
# ╠═66f7ec1d-a5a8-41df-b64e-80634747b7a3
# ╟─d2c92e1d-a3cb-4070-8f2c-cf01e943ad45
# ╠═13b9dc04-a1ce-4aaf-8a14-f92a7c6b5da7
# ╟─cda14c79-7a39-4ef5-8c59-1dd2f0fac202
# ╟─ad534796-d5a1-4c45-b362-68754db23d85
# ╟─44609196-e3f2-446b-9ca9-d8741c43fb05
# ╠═f50f0928-7598-4e36-9a65-876a42b8982e
# ╠═2febee75-48eb-4226-acaf-f33b36eb19bc
# ╟─d16daa50-aff5-447c-8aaf-18da9b57301d
# ╟─454fcd33-52c0-48d8-ba15-ffb5b0249ac4
# ╟─bbc8bc84-98db-4b0d-b50a-182ed6208027
# ╠═48fbc53f-4a01-4996-9761-063b5a5c8aa5
# ╠═c863a3d0-e09b-4cc8-8b32-1b66a7ca78ec
# ╠═e294887b-e9d6-41fb-91e7-61e564c18745
# ╠═5f52d1af-6866-4c51-9eb2-a8eb7e0043d9
# ╠═1b0b424e-1abe-4391-bc36-7d262ff9eaf9
# ╠═9d9b793c-71db-4273-b452-89f5ff937581
# ╠═5f70891f-c990-44f1-9092-fc134d2903b5
# ╠═ccba1495-d81d-4414-8a0a-dcd027ec34fd
# ╟─d5cca937-ffbc-4d78-853b-1379b92b91c2
# ╟─4013f2d2-7324-46c8-a186-46d2855237ee
# ╟─1935b2b0-84c7-4e7c-80f1-e331776a3836
# ╟─c7b00672-4602-4452-a8af-a4918f4a30dc
# ╟─0cf0cb00-7007-42be-a5b7-f06f6c6f3992
# ╟─3c932fb9-8667-475c-841f-2a4d4aa4c3ae
# ╟─b2371496-9da4-4c59-8efe-828844bc91a3
# ╠═bf1571b3-9840-4bb0-a623-ccc53b49c177
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
