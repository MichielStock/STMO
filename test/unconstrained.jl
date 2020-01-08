@testset "unconstrained" begin

    P = [10 1; 1 5]
    q = [-5, 7]
    r = 2.0

    f(x) = 0.5x' * P * x + q' * x + r
    ∇f(x) = P * x .+ q
    ∇²f(x) = P

    xstar = - P \ q
    x0 = zero(xstar)

    @test gradient_descent(f, copy(x0), ∇f) ≈ xstar
    @test coordinate_descent(f, copy(x0), ∇f) ≈ xstar
    @test newtons_method(f, copy(x0), ∇f, ∇²f) ≈ xstar
end
