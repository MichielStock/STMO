@testset "test functions" begin

    import STMO.TestFuns: ackley, branin, rosenbrock, rastrigine, flower, booth, fquadr, fnonquadr

    for fun in [ackley, branin, rosenbrock, rastrigine, flower, booth]

        @test fun([3, 4]) isa Real
        @test fun([0.0, 0.0]) isa Real
        @test fun(3, 4) isa Real
        @test fun(0.0, 0.0) isa Real
    end
end
