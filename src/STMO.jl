module STMO




using Plots, LinearAlgebra

include("helpers.jl")
export myred, mygreen, myyellow, myblue, myblack, myorange, mycolors
export trace, notrack, PathTrack, nsteps
export path, path!, plotobj, plotobj!, colorscatter
export dist

include("quadratic.jl")
export fquad, quadratic, solve_quadratic, quadratic_ls, gradient_descent, plot_quadratic
export generate_noisy_measurements, make_bookkeeping, make_connection_matrix, signalfun

include("unconstrained.jl")
export gradient_descent, coordinate_descent, newtons_method, backtracking_line_search


include("optimaltransport.jl")
export OptimalTransport

include("testfuns.jl")
export TestFuns

end # module
