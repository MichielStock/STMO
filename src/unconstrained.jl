#=
Created on Monday 6 January 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Functions for unconstrained optimization.
=#

module Unconstrained

import STMO: Tracker, trace, notrack
using LinearAlgebra

"""
    backtracking_line_search(f, x, Δx, ∇f, α::Real=0.1,
                        β::Real=0.7)

Uses backtracking for finding the minimum over a line.

Inputs:
    - f: function to be searched over a line
    - x: initial point
    - Δx: direction to search
    - ∇f: gradient of f
    - α: hyperparameter
    - β: hyperparameter

Output:
    - t: suggested stepsize
"""
function backtracking_line_search(f, x, Δx, ∇f; α::Real=0.1,
                        β::Real=0.7)
    @assert 0 < α < 0.5 && 0 < β < 1 "incorrect values for α and/or β"
    t = 1.0
    while f(x + t * Δx) > f(x) + α * t * ∇f(x)' * Δx
        t *= β
    end
    return t
end

"""
    gradient_descent(f, x₀, ∇f; α::Real=0.2, β::Real=0.7,
        ν::Real=1e-3, tracker::Tracker=notrack)

General gradient descent algorithm.

Inputs:
    - f: function to be minimized
    - x₀: starting point
    - ∇f: gradient of the function to be minimized
    - α: parameter for btls
    - β: parameter for btls
    - ν: parameter to determine if the algorithm is converged
    - tracker: a structure to store the path

Outputs:
    - xstar: the found minimum
"""
function gradient_descent(f, x₀, ∇f; α::Real=0.2, β::Real=0.7,
      ν::Real=1e-5, tracker::Tracker=notrack)

    x = x₀  # initial value
    Δx = similar(x)
    while true
        Δx .= -∇f(x) # choose direction
        if norm(Δx) < ν
            break  # converged
        end
        t = backtracking_line_search(f, x, Δx, ∇f, α=α, β=β)
        x .+= t * Δx  # do a step
        trace(tracker, x)
    end
    return x
end

"""
    coordinate_descent(f, x₀, ∇f; α::Real=0.2, β::Real=0.7,
        ν::Real=1e-3, tracker::Tracker=notrack)

General coordinate descent algorithm.

Inputs:
    - f: function to be minimized
    - x₀: starting point
    - ∇f: gradient of the function to be minimized
    - α: parameter for btls
    - β: parameter for btls
    - ν: parameter to determine if the algorithm is converged
    - tracker: a structure to store the path

Outputs:
    - xstar: the found minimum
"""
function coordinate_descent(f, x₀::Vector, ∇f; α::Real=0.2, β::Real=0.7,
      ν::Real=1e-5, tracker::Tracker=notrack)
    x = x₀  # initial value
    Δx = zero(x)
    ∇fx = similar(x)
    while true
        ∇fx .= ∇f(x)
        i = argmax(abs.(∇fx))   # choose direction
        Δx[i] = -∇fx[i]
        if norm(∇fx) < ν
            break  # converged
        end
        t = backtracking_line_search(f, x, Δx, ∇f, α=α, β=β)  # BLS for optimal step size
        x .+= t * Δx  # do a step
        Δx[i] = 0.0  # reset
        trace(tracker, x)
    end
    return x
end

"""
    newtons_method(f, x₀, ∇f, ∇²f; α::Real=0.2, β::Real=0.7,
        ϵ::Real=1e-7, tracker::Tracker=notrack)

General Newton method.

Inputs:
    - f: function to be minimized
    - x₀: starting point
    - ∇f: gradient of the function to be minimized
    - ∇²f: Hessian of the function to be minimized
    - α: parameter for btls
    - β: parameter for btls
    - ϵ: parameter to determine if the algorithm is converged
    - tracker: (bool) store the path that is followed?

Outputs:
    - xstar: the found minimum

"""
function newtons_method(f, x₀, ∇f, ∇²f; α::Real=0.2, β::Real=0.7,
      ϵ::Real=1e-5, tracker::Tracker=notrack)

    x = x₀  # initial value
    Δx = similar(x)
    ∇fx = similar(x)
    while true
        ∇fx .= ∇f(x)
        Δx .= - ∇²f(x) \ ∇fx # choose direction
        λ² = - (Δx' * ∇fx)  # newton decrement
        if λ² < ϵ
            break  # converged
        end
        t = backtracking_line_search(f, x, Δx, ∇f, α=α, β=β)
        x .+= t * Δx  # do a step
        trace(tracker, x)
    end
    return x
end

export gradient_descent, coordinate_descent, newtons_method, backtracking_line_search

end
