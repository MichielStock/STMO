#=
Created on Thursday 23 July 2020
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Illustration of Particle Swarm Optimization.
=#

# first we define a particle with a position `x` and a velocity `v`

using STMO.TestFuns

fun = TestFuns.ackley
x1lims = (-10, 10)
x2lims = (-10, 10)

mutable struct Particle{T}
    x::T
    v::T
    x_best::T
end

Particle(x) = Particle(x, zero(x), x)

init_population(n_particles, lims...) = [Particle([(u - l) * rand() + l for (l, u) in lims]) for i in 1:n_particles]

"""
    particle_swarm_optimization!(f, population, k_max;
            w=1, c1=1, c2=1, tracker=nothing)

Performs Particle Swarm Optimization to minimize a function `f`. Give an initial
vector of particles (type `Particle`) and the `k_max`, the number of iterations.

Optionally set hyperparameters `w`, `c1` and `c2` (default value of 1).
"""
function particle_swarm_optimization!(f, population::Vector{Particle}, k_max;
        w=1, c1=1, c2=1, tracker=nothing)
    # find best point
    y_best, x_best = minimum((((f(part.x), part.x)) for part in population))
    for k in 1:k_max
        # update population
        for particle in population
            r1, r2 = rand(2)
            particle.v .= w * particle.v + c1 * r1 * (particle.x_best .- particle.x) .+
                c2 * r2 * (x_best .- particle.x)
            particle.x .+= particle.v
            fx = f(particle.x)
            # update current best
            fx < f(x_best) && (particle.x_best .= particle.x)
            # update global best
            if fx < y_best
                y_best = fx
                x_best .= particle.x
            end
        end
        tracker isa Nothing || tracker(population)
    end
    return y_best, x_best
end

population = init_population(50, x1lims, x2lims)
particle_swarm_optimization!(fun, population, 100, w=1)

function pso_animation(f, population, k_max;
        w=1, c1=1, c2=1)
    # find best point
    y_best, x_best = minimum((((f(part.x), part.x)) for part in population))
    objective = [y_best]
    # determine xvals and yvals depending on the spread of the particles
    x1min = minimum((part.x[1] for part in population))
    x1max = maximum((part.x[1] for part in population))
    x2min = minimum((part.x[2] for part in population))
    x2max = maximum((part.x[2] for part in population))
    x1steps = (x1max - x1min) / 200
    x2steps = (x2max - x2min) / 200
    anim = @animate for k in 1:k_max

        swarmplot = heatmap(x1min:x1steps:x1max, x2min:x2steps:x2max,
                        (x,y)->f((x,y)), color=:speed)
        xlims!((x1min, x1max))
        ylims!((x2min, x2max))
        # update population
        for particle in population
            r1, r2 = rand(2)
            particle.v .= w * particle.v + c1 * r1 * (particle.x_best .- particle.x) .+
                c2 * r2 * (x_best .- particle.x)
            particle.x .+= particle.v
            fx = f(particle.x)
            # update current best
            fx < f(x_best) && (particle.x_best .= particle.x)
            # update global best
            if fx < y_best
                y_best = fx
                x_best .= particle.x
            end
            scatter!(swarmplot, [particle.x[1]], [particle.x[2]],
                        color=:red, label="")
        end
        push!(objective, y_best)
        pobj = plot(0:k, objective, label="objective")
        xlabel!("iteration")
        ylabel!("best objective")
        plot(swarmplot, pobj)
    end
    return anim
end

population = init_population(50, x1lims, x2lims)
pso_no_momentum = pso_animation(fun, population, 100, w=1, c1=0.9, c2=0.9)
gif(pso_no_momentum, "figures/pso_no_momentum.gif", fps=4)

population = init_population(50, x1lims, x2lims)
pso_momentum = pso_animation(fun, population, 100, w=0.8, c1=0.9, c2=0.9)
gif(pso_momentum, "figures/pso_momentum.gif", fps=4)
