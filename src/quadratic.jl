module Quadratic

using STMO: Tracker, notrack, PathTrack, trace

export fquad, plot_quadratic, solve_quadratic, quadratic_ls, gradient_descent

using LinearAlgebra, Plots

"""Compute a quadratic function."""
fquad(x::Vector, P::Matrix, q::Vector, r::Real=0.0) = 0.5x' * P * x + q' * x + r

"""Plot a 1-D quadratic function."""
function plot_quadratic(p::Real, q::Real, r::Real, (xl, xu), stepsize=0.1; kwargs...)
    return plot(x -> p*x^2 + q*x+r, xl:stepsize:xu, xlabel="\$x\$"; kwargs...)
end

"""Plot a 2-D quadratic function."""
function plot_quadratic(P::AbstractMatrix, q::AbstractVector, r::Real,
                            (x1l, x1u), (x2l, x2u), stepsize=0.1; kwargs...)
    fun = (x1, x2) -> [x1,x2] |> x -> 0.5x' * P * x + q' * x + r
    return contour(x1l:stepsize:x1u, x2l:stepsize:x2u, fun, xlabel="\$x_1\$",
                                            ylabel="\$x_2\$"; kwargs...)
end


"""
    solve_quadratic(p::Real, q::Real[, r::Real=0.0])

Finds the minimizer of an 1-D quadratic system. If the quadratic term `p` is
negative, the function raises an error.

Inputs:
    - `p`, `q`, `r`: the terms of the 1-D quadratic system

Output:
    - xstar: the minimizer, a number
"""
function solve_quadratic(p::Real, q::Real, r::Real=0.0)
    @assert p > 0.0
    return - q / p
end


"""
    solve_quadratic(P::AbstractMatrix, q::AbstractVector, r::Real=0)

Finds the minimizer of an N-D quadratic system.
P is assumed to be a symmetric positive-definite matrix.

Inputs:
    - P, q, r: the terms of the nD quadratic system

Output:
    - xstar: the minimizer, an (n x 1) vector
"""
function solve_quadratic(P::AbstractMatrix, q::AbstractVector, r::Real=0.0)
    return - P \ q
end


"""
    quadratic_ls(P::AbstractMatrix, q::AbstractVector, Δx::AbstractVector,
                                    x::AbstractVector)

Find the exact step size that minimized a quadratic system in
a given point x for a given search direction Dx

Inputs:
    - P, q: the terms of the nD quadratic system
    - x: starting point
    - Δx: search direction

Output:
    - t: optimal step size
"""
function quadratic_ls(P::AbstractMatrix, q::AbstractVector, Δx::AbstractVector,
                                    x::AbstractVector)
    t = solve_quadratic(Δx' * P * Δx, Δx' * P * x + Δx ⋅ q)
    return t
end

"""
    gradient_descent(P::AbstractArray, q::AbstractVector,
            x₀::AbstractVector; β::Real=0.0, ϵ::Real=1e-6,
            tracker::Tracker=notrack)

Computes the minimizes of a quadratic system using gradient descent. Optionally
provide momentum.

Inputs:
    - P, q: the terms of the nD quadratic system
    - x₀: starting point
    - ϵ: convergence parameter
    - β: momentum parameter
    - tracker: object of the type `Tracker` to save the steps

Outputs:
    - xstar: the found minimum
"""
function gradient_descent(P::AbstractArray, q::AbstractVector,
            x₀::AbstractVector; β::Real=0.0, ϵ::Real=1e-6,
            tracker::Tracker=notrack)
    @assert 0 ≤ β < 1
    x = x₀  # initial value
    Δx = zero(x)  # pre-allocate a vector for the gradient
    while true
        Δx .= (1.0 - β) * (- P * x .- q) .+ β *  Δx
        if norm(Δx) < ϵ
            break
        end
        # determine stepsize using exact line search
        t = quadratic_ls(P, q, Δx, x)
        # perform step
        x .+= t * Δx
        trace(tracker, x)  # saves the steps
    end
    return x
end


# SIGNAL RECOVERY
# ---------------

module SignalRecovery
using LinearAlgebra: I

signalfun(x, n) = 3sin(x * 2 * pi / n) +
            2cos(x * 4 * pi / n) +
            sin(x * 4 * pi / n) + 0.8 * cos(x * 12 * pi / n)

"""
    generate_noisy_measurements(m, n; σ=1.0)

Generate noisy measurements according to some function f

Inputs:
    - m : number of observations
    - n : dim of x
    - σ : normally distributed noise (default = 1)

Output:
    - y : vector of noisy measurements
    - ind : vector of indices
"""
function generate_noisy_measurements(m, n; σ=1.0)
    ind = rand(1:n, m)
    y = signalfun.(ind, n) .+ randn(m) * σ
    return y, ind
end


"""
    make_connection_matrix(n, γ=100)

Generates the kernel matrix and the inverse kernel matrix

Uses γ as a characteristic length scale of the radial basis kernel,
assumes periodic boundaries.

Inputs:
    - n : number of points
    - γ : length scale

Output:
    - K
    - Kinv
"""
function make_connection_matrix(n, γ=100)
    # cyclic distance
    d(i, j) = min((i-j)^2, ((i-n)-j)^2, ((j-n)-i)^2)
    # distance matrix
    D = [d(i, j) for i in 1:n, j in 1:n]
    # kernel matrix
    K = exp.(-D / γ^2) + 1e-2I
    return K, inv(K)
end

"""
    make_bookkeeping(I, n)

Constructs the bookkeeping matrix R

Inputs:
    - ind : the indices of the measurements
    - n : dimensionality of signal vector

Output:
    - R : m x n bookkeeping matrix
"""
function make_bookkeeping(ind, n)
    m = length(ind)
    R = zeros(Int, m, n)
    for (i, j) in enumerate(ind)
        R[i,j] = 1
    end
    return R
end

export generate_noisy_measurements, make_connection_matrix, make_bookkeeping, signalfun

end # module SignalRecovery

export SignalRecovery

end  # module Quadratic
