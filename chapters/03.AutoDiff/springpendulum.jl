using DifferentialEquations
using Plots

function springpendulum(du, u, p, t)
    θ̇, θ, ṙ, r = u
    m, b, l, k, ζ = p
    g = 9.81
    du[1] = (-b * θ̇ - g * r * sin(θ)) / (r^2)
    du[2] = θ̇
    du[3] = (-k * (r - l) - ζ * ṙ + m * g * cos(θ)) / m
    du[4] = ṙ
end


tend = 10.0

u0 = [0, π/3.0, 0, 10.0]
tspan = (0.0, tend)
p = [50.0, 60.0, 10.0, 125.0, 60.0]
m, b, l, k, ζ = p
prob = ODEProblem(springpendulum!,u0,tspan,p)
sol = solve(prob)
plot(sol)
png("solution_de.png")

#=
function eulersolve(fun, u0, tsteps, p)
    u = u0
    solution = copy(u0)
    du = Array{Any}(undef, size(u))
    for (t, Δt) in zip(tsteps[2:end], diff(tsteps))
        fun(du, u, p, t)
        u = u + Δt * du
        solution = hcat(solution, u)
    end
    return solution'
end
=#

function eulersolve(fun, u0, (t0, tend), p, stepsize=0.1)
    u = copy(u0)
    du = similar(u)
    for t in t0:stepsize:tend
        fun(du, u, p, t)
        u .+= stepsize * du
    end
    return u
end

solution = eulersolve(springpendulum!, u0, tsteps, p)

plot(tsteps, solution[:,2], label="angle", lw=2)
plot!(tsteps, solution[:,4], label="r", lw=2)
png("mysolution.png")

traject_from_init(theta0) = eulersolve(springpendulum!, [0.0, theta0, 0.0, l], tspan, p)[[2, 4]]

diff_complexdiff(f, x;h=1e-10) = imag(f(x+h*im))/h
