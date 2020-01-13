@testset "test functions" begin

    import STMO.TestFuns: branin, rosenbrock, rastrigine, flower

    for fun in [branin, rosenbrock, rastrigine, flower, booth]

        @test fun([3, 4]) isa Real
        @test fun([0.0, 0.0]) isa Real
    end
end
