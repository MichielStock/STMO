# STMO
Selected Topics in Mathematical Optimization, now in *Julia*.

If you have *Julia* and *IJulia*-notebooks installed, clone the repo (check the section on github below if you don't know what this means) and work local in the notebooks, this is recommended as it is the only way to save your work. Otherwise click on the badge below to open a Binder session or check the *installation instructions* below to install *Julia* and *IJulia*-notebooks. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MichielStock/STMO/master)

While using Binder is convenient in the short term, it will take a while to start up every time and it will only allow you to follow along with the notebooks, without having the ability to save your work.


## Installation instructions
### Github
Using Git or Github desktop is recommended for this course. In case you don't already have git or github installed, this can be done by following the instructions for your operating system here [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for Git and [here](https://desktop.github.com/) for github desktop. Using git, clone (i.e. download the files of) the course repostitory by typing 
```
git clone https://github.com/MichielStock/STMO.git
```
In the command promt, after navigating to where you want to save the course files.
### Installing Julia
1. Download the *Julia* binaries for your system [here](https://julialang.org/downloads/) we suggest to install the Long-term support release, v1.0.5
2. Check the [Platform Specific Instructions](https://julialang.org/downloads/platform.html) of the official website to install *Julia*

### Installing the STMO package
All required packages for this course are bundled together in the STMO package, which can be installed as follows. 

In julia, enter *package mode* by pressing the "]" key.  All required packages will be installed by then typing (or copying) at the `(v1.2) pkg> ` prompt:
```
add https://github.com/MichielStock/STMO.git
```

### Running the IJulia Notebook
If you are comfortable managing your own Python/Jupyter installation, you can just run `jupyter notebook` yourself in a terminal. To simplify installation, you can alternatively type the following in Julia, at the `julia>` prompt:
```julia
using IJulia
notebook()
```
to launch the IJulia notebook in your browser.

The first time you run `notebook()`, it will prompt you
for whether it should install Jupyter.  Hit enter to have it use the [Conda.jl](https://github.com/Luthaf/Conda.jl) package to install a minimal Python+Jupyter distribution (via [Miniconda](http://conda.pydata.org/docs/install/quick.html)) that is private to Julia (not in your `PATH`).
On Linux, it defaults to looking for `jupyter` in your `PATH` first, and only asks to installs the  Conda Jupyter if that fails; you can force it to use Conda on Linux by setting `ENV["JUPYTER"]=""` during installation (see above).  (In a Debian or Ubuntu  GNU/Linux system, install the package `jupyter-client` to install the system `jupyter`).
[source](https://raw.githubusercontent.com/JuliaLang/IJulia.jl/master/README.md)

