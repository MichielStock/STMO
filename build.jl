#=
Created on Sunday 5 January 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Build all the notebooks and stuff. Based on a file by Bram De Jaegher.
=#

t₀ = time()

#=
using Weave


dirs = ["00.Introduction",
        "01.Brackets",
        "02.Quadratic",
        "03.AutoDiff",
        "04.Unconstrained",
        "05.Constrained",
        "06.OptimalTransport",
        "07.MST",
        "08.ShortestPath",
        "09.NP-Complete",
        "10.Metaheuristics",
        "11.TSP"
        ]


function cleanTemps(filename, dir; exts=[".out", ".log", ".aux", ".fls", ".fdb_latexmk"])
  for ext in exts
    rm(dir * filename * ext)
  end
end

for dir in dirs
  dir = "chapters/$dir/"
  files = readdir(dir)
  filter!(fn -> occursin(".jmd", fn), files)
  for filename in files
    filename = filename[1:end-4]
    println("Building $(filename)...")
    weave(joinpath(dir, filename * ".jmd"); doctype="md2pdf")
    cleanTemps(filename, dir)
    convert_doc(joinpath(dir, filename * ".jmd"), dir * filename * ".ipynb")
  end
end
=#

# FIGURES

include("scripts/maketestfigs.jl")
include("scripts/monge.jl")
include("scripts/unconstrained.jl")
include("scripts/colortransfer.jl")

tₑ = time()

t = Int(round(tₑ - t₀))

println("Building took $(div(t, 60))m$(t % 60)s")
