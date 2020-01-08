@testset "unconstrained" begin

    P = [10 1; 1 5]
    q = [-5, 7]
    r = 2.0

    f(x) = 0.5x' * P * x + q' * x + r
    ∇f(x) = P * x .+ q
    ∇²f(x) = P

    xstar = - P \ q
    x0 = zero(xstar)

    @test isapprox(gradient_descent(f, copy(x0), ∇f, ν=1e-6), xstar, atol=1e-5)
    @test isapprox(coordinate_descent(f, copy(x0), ∇f, ν=1e-6), xstar, atol=1e-5)
    @test isapprox(newtons_method(f, copy(x0), ∇f, ∇²f, ϵ=1e-6), xstar, atol=1e-5)
end
