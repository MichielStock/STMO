@testset "Optimal tranport" begin
    using STMO.OptimalTransport

    C = [1 1 0;
        0 1 1;
        1 0 1]

    @testset "Monge" begin

        @test monge_brute_force(C) == ([3, 1, 2], 0)
        @test monge_brute_force(1.0C) == ([3, 1, 2], 0.0)
    end

    @testset "Sinkhorn" begin
        a, b = [1, 1, 1] / 3, [0.25, 0.6, 0.15]
        Ph = sinkhorn(C, a, b, λ=10, ϵ=1e-10)
        Pl = sinkhorn(C, a, b, λ=10, ϵ=1e-10)

        @test sum(Ph, dims=1)[:] ≈ b
        @test sum(Ph, dims=2)[:] ≈ a
        @test sum(Pl, dims=1)[:] ≈ b
        @test sum(Pl, dims=2)[:] ≈ a

        @test sum(Ph .* C) ≤ sum(Pl .* C)  # lower cost

        h(p) = -sum(p .* log.(p))
        @test h(Ph) ≤ h(Pl)
    end
end
