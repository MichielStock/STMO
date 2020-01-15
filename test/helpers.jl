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
end
