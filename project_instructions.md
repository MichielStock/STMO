# Assignments

Edition 2020-2021

This file gives a detailed overview of what you have to do for this project. 

## In brief

For the exam project, you pick a optimization related topic of your interest that is *not* covered in detail in class. This can be an algorithm, an application you solve with methods seen the course or some theoretical aspect you want to study. You write some code that you add to the STMOZOO codebase (including documentation, tests etc.) and illustrate you application in a notebook.

## Getting started

- [ ] pick a project (take a look at `project ideas.md` or discuss with Michiel)
- [ ] [fork](https://docs.github.com/en/enterprise-server@2.20/github/getting-started-with-github/fork-a-repo) this repo
- [ ] rename your repo using a short indicative name, e.g., `GeneticProgramming.jl`. Add `.jl` to indicate this is a Julia package. **Don't use spaces in the name!**
- [ ] make a local clone of the repository 
- [ ] add the repo with your project to the project sheet
- [ ] update the `readme.md`
  - [ ] add title
  - [ ] add your names
  - [ ] add a small abstract/example of what the code should do

## Source code

Every project needs to have some source code, at least one function! You have to decide which parts belong in the source code (and can hence be readily loaded by other users) and which parts of your project will be in the notebook where people can see and interact with your code.

Developing code can be done in any text editor, though we highly recommend [Visual Studio Code](https://code.visualstudio.com/), with Juno the environment for Julia. [Atom](https://atom.io/) is an alternative but is not supported anymore. When developing, you have to activate your project. Assuming that the location of the REPL is the project folder, open the Pkg manager (typing `]`) and type `activate .`. The dot indicated the current directory. If you use external packages in your project, for example, Zygote or LinearAlgebra, you have to add them using `add PACKAGE` in the package manager. This action will create a dependency and update the `Project.toml` file.

Importantly, all your code should be in a [module](https://docs.julialang.org/en/v1/manual/modules/), where you export only the functions useful for the user.

- [ ] In the `src` folder, add a new Julia file with your source code, for example `geneticprogramming.jl`. Don't use spaces or capitals in the file name.
- [ ] Link your file in `STMOZOO.jl` using `include(filename)`,  running the code.
- [ ] Create a module environment in your file for all your code. Use [camel case](https://en.wikipedia.org/wiki/Camel_case) for the name.
  - use `module GeneticProgramming begin ... end` to wrap your code;
  - import everything you need from external packages: `using LinearAlgebra: norm`;
  - export your functions using `export`
- [ ] write awesome code!
- [ ] take a look at your code regarding the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [ ] check the [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [ ] document *every* function! Make sure that an external user can understand everything! Be liberal with comments in your code. Take a look at the [guidelines](https://docs.julialang.org/en/v1/manual/documentation/)

## Unit tests

Great, we have written some code. The question is, does it work? Likely you have experimented in the REPL. For a larger project, we would like to have guarantees that it works, though. Luckily, this is very easy in Julia, where we can readily use [Unit testing](https://docs.julialang.org/en/v1/stdlib/Test/).

You will have to write a file with some unit tests, ideally testing every function you have written! The fraction of functions that are tested is called [code coverage](https://en.wikipedia.org/wiki/Code_coverage). This project is monitored automatically using Travis (check the button on the readme page!). Currently, coverage is 100%, so help to keep this as high as possible!

Tests can be executed using the `@test` macro. You evaluate some functions and check their results. The result should evaluate to `true`. For example: `@test 1+1 == 2` or `@test √(9) ≈ 3.0`. 

It makes sense to group several tests, which can be done using `@testset "names of tests" begin ... end`.

Your assignments:
- [ ] add a source file to the `test/` folder, the same name as your source code file.
- [ ] add an `include(...)` with the filename in `runtests.jl`
- [ ] in your file, add a block `@testset "MyModule" begin ... end` with a series of sensible unit tests. Use subblocks of `@testset` if needed.
- [ ] run your tests, in the package manager, type `test`. It will run all tests and generate a report.

Travis will automatically run your unit tests online when you push to the origin repo.

## Documentation

> **This part is optional!** If you just provide a couple of examples how to use your code in the readme, it is fine. 

Hopefully, you have already documented all your functions, so this should be a breeze! We will generate a documentation page using the [Documenter](https://juliadocs.github.io/Documenter.jl/stable/man/guide/) package. Since we will not put the project in the package manager, we won't host the documentation, though we generate HTML pages anyway.

- [ ] add markdown file to `docs/src/man` with the documentation.
- [ ] write a general introduction explaining the rationale of your code.
- [ ] use a `@docs` block to add your functions with their documentation.
- [ ] update the `make.jl` file, linking your page.
- [ ] run the `make.jl` file to generate the documentation, an HTML file, not added to the repo.

## Notebook

Finally, you have to add a [Pluto](https://github.com/fonsp/Pluto.jl) notebook to the `notebook` folder. Again use the same name you used for your source code. Depending on the nature of your project, this will be the most extensive task! Make full use of Pluto's interactivity to illustrate your code. In contrast to the documentation page, this is not the place to explain your functions but rather show what you can do with your software or explain a concept.

Alternatively, you may use [Literate](https://fredrikekre.github.io/Literate.jl/v2/) to have script with text annotation to explain your code/package. Up to you what you feel most comfortable with.

## Code review

Each of you will have to perform a code review of two other projects. You have till noon 13h of the exam date to do this, though it should not take too long. The aim is to **help** the other groups to make each other's project even better. 

- [ ] make a fork or local clone of the repo of the person you are reviewing;
- [ ] check the source code, is the documentation clear? Anything obvious that can be improved.
- [ ] run the tests. Do they work? Anything that could be tested but is not done so?
- [ ] Is the documentation clear? Do you find any typos? Could an example be added?
- [ ] Take a look at the notebook. Any suggestions there to improve this?

Big things can be addressed by opening an issue. Small fixes and suggestions to the other person's code can be done immediately and via a pull request.

Afterwards, you have the rest of the day to:
- [ ] merge the entire request and fix any issues you find meaningful.
- [ ] fill in a small questionnaire on Ufora about your project and the projects you have reviewed.

When your code is final, you can [tag](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/managing-commits/managing-tags) your latest commit and mention Michiel you are finished (mention `@michielstock`).
