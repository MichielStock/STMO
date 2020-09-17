using DiffEqFlux
using Plots

function autocatlysis!(du, u, (a, k₁, k₋₁), t)
    du .= k₁ * a * u - k₋₁ * u.^2
end

p = (10.0, 0.1, 0.01)

u0 = [0.1]
tspan = (0.0, 10.0)
a, k₁, k₋₁ = p
prob = ODEProblem(autocatlysis!,u0,tspan,p)
sol = solve(prob)
plot(sol)
