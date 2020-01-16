#=
Created on Thurday 16 Jan 2020
Last update:

@author: Michiel Stock
michielfmstock@gmail.com

Solution of the Monge problem
=#

using STMO
using STMO.OptimalTransport
using Plots, LaTeXStrings

p = scatter(X1[:,1], X1[:,2], color=myorange, label="location cells at t1")
xlabel!("\$x\$")
ylabel!("\$y\$")
scatter!(X2[:,1], X2[:,2], color=mygreen, label="location cells at t2")

C = dist(X1, X2)

perm, cost = monge_brute_force(C)

for (i, j) in enumerate(perm)
    plot!(p, [X1[i,1], X2[j,1]], [X1[i,2], X2[j,2]], label="", color=myred,
                alpha=0.8, lw=2)
end

title!(p, "Monge solution with cost=$cost")

savefig(p, "figures/mongesol.png")
