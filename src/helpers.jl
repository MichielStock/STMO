#=
Created on Thurday 02 January 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

General stuff.
=#

# COLORS
# ------

myblue = "#264653"
mygreen = "#2a9d8f"
myyellow = "#e9c46a"
myorange = "#f4a261"
myred = "#e76f51"
myblack = "#50514F"

mycolors = [myblue, myred, mygreen, myorange, myyellow]


# TRACING
# -------


abstract type Tracker end

struct NoTrack <: Tracker
end

notrack = NoTrack()

struct PathTrack{T<:Any} <: Tracker
    xsteps::Array{T,1}
    PathTrack(x::T) where {T} = new{T}(copy(T[x]))
end

trace(::NoTrack, x) = nothing
trace(tracker::PathTrack, x) = push!(tracker.xsteps, copy(x))

nsteps(tracker::PathTrack) = length(tracker.xsteps) - 1

function getx(tracker::PathTrack)
    n = length(tracker.xsteps)
    return [x[1] for x in tracker.xsteps], [x[2] for x in tracker.xsteps]
end

# PLOTS
# -----

path(tracker::PathTrack; kwargs...) = plot(getx(tracker)...; color=mygreen, lw=2, kwargs...)
path!(tracker::PathTrack; kwargs...) = plot!(getx(tracker)...; color=myorange, lw=2, kwargs...)

plotobj(f::Function, tracker::PathTrack; kwargs...) = plot(0:nsteps(tracker), f.(tracker.xsteps),
                                                 lw=2, color=myorange, xlabel="iteration"; kwargs...)
plotobj!(f::Function, tracker::PathTrack; kwargs...) = plot!(0:nsteps(tracker), f.(tracker.xsteps);
                                                 lw=2,color=mygreen, kwargs...)
