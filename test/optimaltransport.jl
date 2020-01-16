@testset "Optimal tranport" begin
    using STMO.OptimalTransport
    
    @testset "Monge" begin
        C = [1 1 0;
            0 1 1;
            1 0 1]
        @test monge_brute_force(C) == ([3, 1, 2], 0)
        @test monge_brute_force(1.0C) == ([3, 1, 2], 0.0)
    end
end
