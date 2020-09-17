---
author: dimiboeckaerts
---

# Let's get going with Julia

Let's get going with Julia! This summary-based guide into programming with Julia is meant for students and researchers with a general background in programming but no specific knowledge of Julia. The first two sections get you up and running in Julia, while the third and fourth section dive into some more (bio)engineering-oriented material. Finally, the last section provides additional resources to explore more from Julia. Happy learning!

[TOC]


## The basics

What is Julia again?
https://julialang.org/learning/getting-started/

Why should you learn Julia?
https://blog.goodaudience.com/10-reasons-why-you-should-learn-julia-d786ac29c6ca

Let's download & install Julia. You can either download the language itself and engage with it through the command-line (or notebooks, see below):
https://julialang.org/downloads/
or you can download JuliaPro (requires a free account) which comes with Juno, an IDE to work in:
https://juliacomputing.com/products/juliapro.html

First baby steps around the block:
https://docs.julialang.org/en/v1/manual/getting-started/

Learn the basics of Julia:
https://juliabyexample.helpmanual.io/
https://learnxinyminutes.com/docs/julia/

Learn more about types, multiple dispatch and structs:
https://towardsdatascience.com/how-to-learn-julia-when-you-already-know-python-641ed02b3fa7


## Notebooks & Visualization

### Jupyter notebook & IJulia
Coupling Julia to Jupyter notebooks is done with IJulia:
https://www.youtube.com/watch?v=oyx8M1yoboY

### Pluto
Pluto is a lightweight, reactive notebook for Julia, with a lot of useful features:
https://github.com/fonsp/Pluto.jl

### Visualization in Julia
Plots are really straightforward in Julia:
https://docs.juliaplots.org/latest/tutorial/
http://ucidatascienceinitiative.github.io/IntroToJulia/Html/PlotsJL


## Specific topics of interest

### Optimization

Optimization in Julia is organized through projects in the JuMP and JuliaDiff communities, which coordinate the development of a wide breadth of functionality in mathematical programming, optimization and Operations Research. For more details, see the STMO course (https://github.com/MichielStock/STMO)!

Julia, JuMP and JuliaDiff packages address a variety of optimization problem categories (e.g. linear, nonlinear, semidefinite, second-order conic, and mixed-integer), provide functionality for advanced acceleration techniques (automatic differentiation and dual-number calculations), as well as provide high-level, embedded, domain-specific algebraic modeling languages directly in Julia (source: juliacomputing.com)

Some fun examples with JuMP:
https://www.juliaopt.org/notebooks/JuMP-Sudoku.html
https://www.juliaopt.org/notebooks/JuMP-Rocket.html

JuMP overview & hands-on tutorial:
https://www.youtube.com/watch?v=7LNeR299q88
https://github.com/mfalt/juliacourse/blob/master/lecture6/JuMP_Overview.ipynb

A detailed overview of optimization in Julia in the book Julia Programming for Operations Research:
https://www.softcover.io/read/7b8eb7d0/juliabook/frontmatter

Another family of packages for non-linear optimization in Julia is JuliaNLSolvers:
https://github.com/JuliaNLSolvers

### Bioinformatics & computational biology

A good documentation on Julia tools for bioinformatics and computational biology is hard to find at first. BioJulia (the BioPython of Julia) seems like the only and all-encompassing Julia library for these kinds of tasks.
https://github.com/BioJulia
https://www.youtube.com/watch?v=6CpPd6tkokQ
- BioSequences.jl to work with DNA & protein sequences
- BioAlignments.jl to do alignments
- BioTools.jl to BLAST
- FASTX.jl to read & write FASTA & FASTQ files
- Phylogenies.jl to do phylogenetics
- ...

Another framework for some bioinformatics-related tasks in Julia is the MiToS package. It handles multiple sequence alignments, protein structures, alignments from Pfam and more.
https://github.com/diegozea/MIToS.jl

Thirdly, ProteinEnsembles is a package that creates ensembles of protein structures:
https://github.com/jgreener64/ProteinEnsembles.jl

Fourthly, simulate DNA sequencing experiments with Pseudoseq.jl:
https://github.com/bioinfologics/Pseudoseq.jl

You can also do regular expressions in Julia:
https://docs.julialang.org/en/v1/manual/strings/#Regular-Expressions
also see: https://juliabyexample.helpmanual.io/#String-Manipulations

Here's an introductory video on general data analysis with Data Frames:
https://www.youtube.com/watch?v=XRClA5YLiIc

Some interesting case studies:
- RNA sequencing data analysis: https://juliacomputing.com/case-studies/rna.html
- Medical diagnosis with AI: https://juliacomputing.com/case-studies/contextflow.html
- Precision medicine: https://juliacomputing.com/case-studies/pathbio.html

### Machine learning

For starters, Scikit-learn has a Julia implementation!
https://github.com/cstjean/ScikitLearn.jl

MLJ is another extensive framework for ML models in Julia that appears to comprise Scikit-learn:
https://github.com/alan-turing-institute/MLJ.jl

Tutorials (on general data science & MLJ):
https://alan-turing-institute.github.io/DataScienceTutorials.jl/
https://www.youtube.com/watch?v=ByFglWPqNlg

### Deep learning

There are quite a few deep learning frameworks to work with in Julia:
https://analyticsindiamag.com/top-9-machine-learning-frameworks-in-julia/

Flux and TensorFlow are two of the main DL frameworks. Knet is a third popular framework.
https://juliacomputing.com/industries/ml-and-ai.html

An overview of how DL is fresh in Julia with tutorial:
https://www.youtube.com/watch?v=4sOQN6cLqEc


## Advanced stuff
Parallel & distributed computing in Julia:
https://www.youtube.com/watch?v=JoRn4ryMclc
https://github.com/mfalt/juliacourse/blob/master/lecture7/distributed.pdf

GPU computing in Julia:
https://www.youtube.com/watch?v=7Yq1UyncDNc

Performance and profiling in Julia:
https://github.com/mfalt/juliacourse/blob/master/lecture3/performance.pdf
https://github.com/crstnbr/JuliaWorkshop19/blob/master/1_One/3_specialization.ipynb


## Where to get additional help?

A Julia (vs Python vs MATLAB) cheatsheet! https://cheatsheets.quantecon.org/

Free courses at Julia Academy: https://juliaacademy.com/courses

Community: https://julialang.org/community/

Books: https://julialang.org/learning/books/

