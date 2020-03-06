@testset "distance" begin
    x = [3, 4]
    y = [0.0, 0.0]

    @test dist(x, y) ≈ 5

    X = [x y]'
    Y = [y x x]'

    D = dist(X, Y)
    @test size(D) == (2, 3)
    @test D[1, 1] ≈ 5
    @test D[1, 3] ≈ 0

    @test size(dist(Y, Y)) == (3, 3)

    @test hamming("banana", "banaan") == 2 
end


@testset "graph" begin

    @test [(1, 2), (3, 3)] isa EdgeList
    @test !([(1, 2), (3, 3, 4)] isa EdgeList)
    edges = [(0.1, 3, 2), (2.0, 1, 2)]
    @test edges isa WeightedEdgeList

    adjlist = edges2adjlist(edges)
    @test adjlist isa AdjList{Float64, Int}
    @test (2.0, 1) in adjlist[2]
    @test (2.0, 1) ∉ edges2adjlist(edges, double=false)[2]

    edgesrec = adjlist2edges(adjlist)
    @test (0.1, 3, 2) in edgesrec && (0.1, 2, 3) in edgesrec

    @test nvertices(adjlist) == nvertices(edges) == 3
    @test vertices(adjlist) == vertices(edges) == Set([1, 2, 3])
    @test isconnected(adjlist) == isconnected(edges) == true
    @test !isconnected([(0.1, 1, 2), (0.1, 2, 3), (0.01, 5, 4)])


end
