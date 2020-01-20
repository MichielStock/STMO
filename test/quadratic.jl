
@testset "Quadratic" begin
    using STMO.Quadratic
    
    p, q, r = 4.0, 3.0, 1.0

    @test solve_quadratic(p, q, r) ≈ -q / p
    @test_throws AssertionError solve_quadratic(-3, 1)

    P = [4 1; 1 2.0]
    q = [-3.0, 1.0]

    @test solve_quadratic(P, q) ≈ - P \ q
    @test norm(P * solve_quadratic(P, q) + q) < 1e-7

    x0 = zeros(2)
    @test gradient_descent(P, q, copy(x0), ϵ=1e-12) ≈ solve_quadratic(P, q)
    tracker = PathTrack(x0)

    @test tracker isa PathTrack
    gradient_descent(P, q, copy(x0), tracker=tracker)


    @test nsteps(tracker) > 1

    @test gradient_descent(P, q, copy(x0), ϵ=1e-12, β=0.2) ≈ solve_quadratic(P, q)
end
