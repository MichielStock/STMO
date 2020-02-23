@testset "shortest path" begin
    using STMO
    using STMO.ShortestPath

    graph = Dict('A' => [(0.5, 'B'), (1.5, 'H')],
        'B' => [(1.0, 'C')],
        'C' => [(1.0, 'D'), (1.0, 'E')],
        'D' => [(2.0, 'F')],
        'E' => [(1.0, 'G')],
        'F' => [(0.1, 'E')],
        'G' => [(1.0, 'F')],
        'H' => [(0.9, 'E')])

    distances, previous = dijkstra(graph, 'A')

    @test distances['D'] ≈ 2.5
    @test previous['E'] == 'H'

    path = reconstruct_path(previous, 'A', 'H')
    @test first(path) == 'A'
    @test last(path) == 'H'

    path_d, cost_AH = dijkstra(graph, 'A', 'H')
    @test path_d == path
    @test cost_AH ≈ distances['H']

    path_as, cost_as = a_star(graph, 'A', 'H', x -> 0.0)
    @test path_as == path
    @test cost_as ≈ distances['H']


end
