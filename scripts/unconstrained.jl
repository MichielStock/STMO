#=
Created on Saturday 22 Feb 2020
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Example plots convergence unconstrained convex optimization.
=#

using STMO
using Plots, LaTeXStrings
using LinearAlgebra
using STMO.TestFuns
using STMO.Unconstrained

# starting point
x0q = [9.0, 2.0];
x0nq = [-1.0, 0.75];

# GRADIENT DESCENT
trackerqgd = PathTrack(x0q);
xstarq = Unconstrained.gradient_descent(fquadr, copy(x0q), grad_fquadr, tracker=trackerqgd);

trackernqgd = PathTrack(x0nq)
xstarnq = gradient_descent(fnonquadr, copy(x0nq), grad_fnonquadr,
                tracker=trackernqgd);

# COORDINATE DESCENT

trackerqcd = PathTrack(x0q);
xstarq = Unconstrained.coordinate_descent(fquadr, copy(x0q), grad_fquadr,
                tracker=trackerqcd);

trackernqcd = PathTrack(x0nq);
xstarq = Unconstrained.coordinate_descent(fnonquadr, copy(x0nq), grad_fnonquadr,
            tracker=trackernqcd);

# NEWTON

trackerqnm = PathTrack(x0q)
xstarq = Unconstrained.newtons_method(fquadr, copy(x0q), grad_fquadr, hess_fquadr, tracker=trackerqnm)

trackernqnm = PathTrack(x0nq)
xstarq = Unconstrained.newtons_method(fnonquadr, copy(x0nq), grad_fnonquadr, hess_fnonquadr, tracker=trackernqnm)

# PLOTTING

pq = contour(-10:0.1:10, -5:0.1:5, (x1, x2) -> fquadr((x1, x2)), xlabel="\$ x_1 \$",
                ylabel="\$ x_2 \$", title="quadratic", fill=false)
path!(trackerqgd, label="GD")
path!(trackerqcd, label="CD", color=myblue)
path!(trackerqnm, label="Newton", color=mygreen)

savefig(pq, "figures/path_quadr.png")

pnq = contour(-2:0.1:2, -1:0.1:1, (x1, x2) -> fnonquadr((x1, x2)), xlabel="\$ x_1 \$",
                ylabel="\$ x_2 \$", title="non-quadratic", fill=false)
path!(trackernqgd, label="GD")
path!(trackernqcd, label="CD", color=myblue)
path!(trackernqnm, label="Newton", color=mygreen)
savefig(pnq, "figures/path_nonquadr.png")

# convergence

fnqexact = newtons_method(fnonquadr, copy(x0nq), grad_fnonquadr, hess_fnonquadr, ϵ=1e-12) |> fnonquadr
nonquadrerr = x -> abs(fnonquadr(x) .- fnqexact);


p1 = plotobj(fquadr, trackerqgd, label="GD", yscale=:log10, ylabel="\$ f(x) - f(x^\\star)\$", title="quadratic", ylim=(1e-15, 10));
plotobj!(fquadr, trackerqcd, color=myblue, label="CD");
plotobj!(fquadr, trackerqnm, color=mygreen, label="Newton");
p2 = plotobj(nonquadrerr, trackernqgd, label="GD", yscale=:log10, ylabel="\$ f(x) - f(x^\\star)\$", title="non-quadratic", ylim=(1e-15, 10));
plotobj!(nonquadrerr, trackernqcd, color=myblue, label="CD");
plotobj!(fquadr, trackernqnm, color=mygreen, label="Newton");
pconv = plot(p1, p2);
savefig(pconv, "figures/unconstrained_conv.png")
