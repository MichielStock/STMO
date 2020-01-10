#=
Created on Sunday 5 January 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Build all the notebooks and stuff. Based on a file by Bram De Jaegher.
=#

using Weave


dirs = ["01.Brackets",
        "02.Quadratic",
        "03.Unconstrained",
        "04.Constrained",
        #"05.AutoDiff"
        ]


function cleanTemps(filename, dir; exts=[".out", ".log", ".aux"])
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
    weave(dir * filename * ".jmd"; doctype="md2pdf")
    cleanTemps(filename, dir)
    convert_doc(dir * filename * ".jmd", dir * filename * ".ipynb")
  end
end
