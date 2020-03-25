#=
Created on Wednesday 18 March 2020
Last update: Tuesday 23 March 2020

@author: Michiel Stock
michielfmstock@gmail.com

This is an illustration of solving the Krontrovich formulation of the optimal
transport problem. It also demonstrates the use of Convex.jl, a package for
convex optimization problems.
=#

using Convex, SCS

# first give the problem parameters, we use the dessert problem

# portions per person
a = [3.0, 3, 3, 4, 2, 2, 2, 1]
# quantities of each dessert
b = [4.0, 2, 6, 4, 4]

# should both have the same sum
@assert sum(a) ≈ sum(b)

preferences = [2 2 1 0 0;
              0 -2 -2 -2 2;
              1 2 2 2 -1;
              2 1 0 1 -1;
              0.5 2 2 1 0;
              0 1 1 1 -1;
             -2 2 2 1 1;
              2 1 2 1 -1]

C = -preferences

n, m = size(C)

P = Variable(n, m)

problem = minimize(sum(P .* C),  # objective
                    P >= 0,  # non-neg constraints
                    P * ones(m) == a,  # row marginals
                    P' * ones(n) == b)  # column marginals

solve!(problem, () -> SCS.Optimizer(verbose=false))

pstar = problem.optval  # best objective
Pstar = P.value  # distribution

using StatsPlots

groupedbar(Pstar, bar_position = :stack, title="Optimal distribution desserts\nobjective = $pstar")

savefig("figures/kantovorisc.png")
