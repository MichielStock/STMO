### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 47f98a52-f2b2-11ea-0ac6-9b4b74b7b15d
using Plots, PlutoUI

# ╔═╡ 5ac4989c-f2b3-11ea-3e8a-a79a4117e48c
using SymEngine

# ╔═╡ 6fe82e5a-f2b8-11ea-101d-d1ee5f9bd5b1
using ForwardDiff

# ╔═╡ 963c4910-f2b8-11ea-12bb-bb42539b1c93
using Zygote

# ╔═╡ 08cf2a80-f2b2-11ea-132d-2f26c934bbb2
md"""
# Numeric and automatic differentiation

STMO

**Michiel Stock**

## Motivation

Up to now, we confidently assumed that we would always be able to compute the derivative or gradient of any function. Despite differentiation being a relatively easy operation, it is frequenty not feasible (or desirable) to compute this by hand. *Numerical differentiation* can provide approximations of th derivate or gradient at a particular point. *Automatic differentiation* directly manipulates the computational graph to generate a function that computes the (exact) derivate. Such methods have advanced greatly in the last years and it is no exageration that their easy use in popular software libraries such as TenserFlow and PyTorch are a cornerstone of deep learning and other machine learning and scientific computing fields.

"""

# ╔═╡ 5154a672-f2b2-11ea-1280-3ba5dadff7f9
md"""
## Definition of a derivative

$$\frac{\text{d}f(x)}{\text{d}x} = f'(x) = \lim _{h\to 0}{\frac {f(x+h)-f(x)}{h}}.$$

Derivation is in essence a mechanical process, following the rules below.

![](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/derivatives.jpeg)

![](https://imgs.xkcd.com/comics/differentiation_and_integration.png)

When we work with function of several variables, we use *partial derivatives* (e.g. $\frac{\partial f(x, y)}{\partial x}$), indicating we keep all variables but $x$ fixed.
"""

# ╔═╡ f53fbf24-f2b2-11ea-2f59-d943884fee9d
md"""
Our running example:

$$f(x) = \log x + \frac{\sin x}{x}$$
"""

# ╔═╡ e6a55456-fe6f-11ea-328c-47fedb388194
md"**Assignment**: implement this function."

# ╔═╡ 2be8656c-f2b3-11ea-2244-5740c807deff
f(x) = log(x) + sin(x) / x

# ╔═╡ 2e00ef04-f2b3-11ea-1960-8136321c71db
plot(f, 1, 5, label="f(x)")

# ╔═╡ f7cd052e-f2bb-11ea-239a-47966e96e909
md"We will evaluate this function in `a`=$2."

# ╔═╡ 7244eba2-f2b3-11ea-045f-ffd208852eda
a = 2.0

# ╔═╡ 401df7a4-f2b3-11ea-12a1-e9b46bcfcf1b
md"""
## Symbolic differentiation

Computing derivatives, as you have seen in basic calculus courses.

By hand or automatically:
- Maple
- Sympy (python)
- Mathematica
- Maxima

Differentiation is *easy* compared to *integration* or *sampling*.

Advantages:
- exact derivatives!
- gives the formula for different evaluations.
    - insight in the system
    - in some cases, closed-form solution extrema by solving $\frac{\text{d}f(x)}{\text{d}x}=0$
- no hyperparameters or tweaking: just works!

Disadvantages:
- some software not flexible enough (gradients, arrays, for-loops,...)
- sometimes explosion of terms: *expression swell*
- not always numerically optimal!
"""

# ╔═╡ 5db7350a-f2b3-11ea-293d-ffc572e0dbe8
@vars x  # define variable

# ╔═╡ d5edd644-fe7c-11ea-2aa8-c772ea2f1bde
f(x)

# ╔═╡ 64b28e4a-f2b3-11ea-0fc0-7d37b4794686
df = diff(f(x), x)

# ╔═╡ 684ff7f4-f2b3-11ea-14f7-7d3f329ecd79
df(a)

# ╔═╡ 2ca60bca-f2b4-11ea-0fec-c50f4e306536
true_diff = df(a);  # save for checking

# ╔═╡ 7a6a18de-f2b3-11ea-1b3d-ef7a2b1b0434
begin
plot(f, 1, 10, label="\$f(x)\$", xlabel="\$x\$", lw=2, color=:green)
plot!(df, 1, 10, label="\$f'(x)\$", lw=2, color=:orange)
end

# ╔═╡ 913234a2-f2b3-11ea-3501-e3bfc54f91fe
md"""
# Numerical differentiation

Finite difference approximation of the derivative/gradient based on a number of function evaluations.

Often based on the limit definition of a derivative. Theoretical analysis using Taylor approximation:

$$f(x + h) = f(x) + \frac{h}{1!}f'(x) + \frac{h^2}{2!}f''(x) + \frac{h^3}{3!}f^{(3)}(x)+\ldots$$

**Forward difference**

$$f'(x)\approx \frac{f(x+h) - f(x)}{h}$$

**Central difference**

$$f'(x)\approx \frac{f(x+h) - f(x-h)}{2h}$$

**Complex step method**

$$f'(x)\approx \frac{\text{Im}(f(x +ih))}{h}$$
"""

# ╔═╡ 0238579a-fe70-11ea-0968-0f11ae9557db
md"**Assignment**: Implement these functions."

# ╔═╡ eef066e0-f2b3-11ea-3cfa-5f35ebf13f76
diff_fordiff(f, x; h=1e-10) = (f(x + h) - f(x)) / h

# ╔═╡ f2158710-f2b3-11ea-16c9-dded5e1747d4
diff_centrdiff(f, x; h=1e-10) = (f(x + h) - f(x - h)) / 2h

# ╔═╡ f54ee2b2-f2b3-11ea-24e8-f556ab9e3b85
diff_complstep(f, x; h=1e-10) = imag(f(x + im * h)) / h

# ╔═╡ 08a565e0-f2b4-11ea-2ab6-f5f3a1b3c22d
diff_fordiff(f, a)

# ╔═╡ 1055a996-f2b4-11ea-14d2-6116cbbaa6ba
diff_centrdiff(f, a)

# ╔═╡ 13db037a-f2b4-11ea-2a38-bd7c1c60f7d9
diff_complstep(f, a)

# ╔═╡ 4f42234e-f2b4-11ea-2c86-93449b8dc60b
md"""
## Intermezzo: floats

Real numbers are always represented as floating point numbers in a computer.

![Encoding of a real number using a `Float32`.](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/floats.png)

By default, Julia uses double precision floats (`Float64`). For brevity, let us take a look at the bit representation of a float. We use `Float32` for brevity's sake.
"""

# ╔═╡ 6adbc7fe-f2b4-11ea-1861-b398d517edfa
num = Float32(10.789)

# ╔═╡ 6cd1accc-f2b4-11ea-0077-e7685b77b1be
bitstring(num)

# ╔═╡ 77d4fd72-f2b4-11ea-0bd6-97fc2299ca2f
md"The first bit encodes the *sign*, here positive."

# ╔═╡ 773af248-f2b4-11ea-154a-cb67beebed3d
sign(num)

# ╔═╡ 8a5ee764-f2b4-11ea-23f3-27664d910c16
md"The next eight bits specify the *exponent*, the magnitude of the number."

# ╔═╡ 8ded9128-f2b4-11ea-0f44-9321077205de
exponent(num)

# ╔═╡ c3a1abd8-f2b4-11ea-0509-a5296ae8b361
md"While the final 23 bits specify the *mantissa*, a number between $[1,2]$ representing the precision."

# ╔═╡ c9f92e66-f2b4-11ea-322f-7778f149453d
significand(num)

# ╔═╡ cfb07cc6-f2b4-11ea-08a7-dd0b13d15b5e
md"These can be used to reconstrunct the number."

# ╔═╡ d56b74f2-f2b4-11ea-12b4-5f54d4999b38
sign(num) * significand(num) * 2^exponent(num)

# ╔═╡ d9f2d114-f2b4-11ea-32b0-43fa1a872a97
md"The *machine precision* of a number can be retained using `eps`. This is the relative error."

# ╔═╡ de1fd700-f2b4-11ea-36dd-4bfba5ecc134
eps(1.2)

# ╔═╡ df14ce72-f2b4-11ea-307b-8fac2889681b
eps(1.2e10)

# ╔═╡ e584f976-f2b4-11ea-0245-67f6bcc4be0e
eps(1.2e-10)

# ╔═╡ eb68674c-f2b4-11ea-0cc2-f792d17a5005
md"""
This brings us with numerical issues we might encounter using numerical differentiation.

**First sin of numerical analysis**:

> *thou shalt not add small numbers to big numbers*

**second sin of numerical analysis**:

> *thou shalt not subtract numbers which are approximately equal*
"""

# ╔═╡ f4edad2c-f2b4-11ea-05fb-3d43c18e9d5c
md"### Back to numerical differentiation"

# ╔═╡ fd9ed266-f2b4-11ea-3c53-3bd2fe30a7f1
begin
	fexamp(x) = 64x*(1-x)*(1-2x)^2*(1-8x+8x^2)^2
	dfexamp = diff(fexamp(x), x)
	error(diff, h; x=1.0) = max(abs(Float64(dfexamp(x)) - diff(fexamp, x, h=h)), 1e-50)
	stepsizes = map(t->10.0^t, -20:0.1:-1);
	plot(stepsizes, error.(diff_fordiff, stepsizes), label="forward difference",
    xscale=:log10, yscale=:log10, lw=2, legend=:bottomright, color=:blue)
	plot!(stepsizes, error.(diff_centrdiff, stepsizes), label="central difference", 		lw=2, color=:red)
	plot!(stepsizes, error.(diff_complstep, stepsizes), label="complex step", lw=2,
            color=:yellow)
	xlabel!("\$h\$")
	ylabel!("absolute error")
end

# ╔═╡ 4e6b2b04-f2b5-11ea-0c52-5f32438e8c68
md"""
Advantages of numerical differentiation:
- easy to implement
- general, no assumptions needed

Disadvantages:
- not numerically stable (round-off errors)
- not efficient for gradients ($\mathcal{O}(n)$ evaluations for $n$-dimensional vectors)
"""

# ╔═╡ 55ff7230-f2b5-11ea-2898-db5e21c764fd
md"""## Approximations of multiplications with gradients

**Gradient-vector approximation**

$$\nabla f(\mathbf{x})^\intercal \mathbf{d} \approx \frac{f(\mathbf{x}+h\cdot\mathbf{d}) - f(\mathbf{x}-h\cdot\mathbf{d})}{2h}$$

**Hessian-vector approximation**

$$\nabla^2 f(\mathbf{x}) \mathbf{d} \approx \frac{\nabla f(\mathbf{x}+h\cdot\mathbf{d}) - \nabla f(\mathbf{x}-h\cdot\mathbf{d})}{2h}$$
"""

# ╔═╡ 688a1f40-f2b5-11ea-3fa8-ebb735323ddd
grad_vect(f, x, d; h=1e-10) = (f(x + h * d) - f(x - h * d)) / (2h)

# ╔═╡ 6d6da8ba-f2b5-11ea-1064-df27ecea7d13
dvect = randn(10) / 10

# ╔═╡ 6fd08e42-f2b5-11ea-36e6-d7ff10412e2b
xvect = 2rand(10)

# ╔═╡ 9a87095e-f2b5-11ea-334e-57a637e20c43
A = randn(10, 10) |> A -> A * A' / 100

# ╔═╡ 67e9fa88-ff03-11ea-1177-c34331398d6c
md"$$g(\mathbf{x}) = \exp(-\mathbf{x}^\intercal A\mathbf{x})$$"

# ╔═╡ 75ae48fe-f2b5-11ea-0614-912091cbfb6a
g(x) = exp(- sum(x .* (A * x)))

# ╔═╡ 8a738272-f2b5-11ea-2de6-f9ae6b30a466
md"Correct gradient and Hessian (by hand)"

# ╔═╡ 7b95d73c-f2b5-11ea-1b29-858f7b6c99bf
∇g(x) = -2g(x) * A * x

# ╔═╡ 836ca94a-f2b5-11ea-07e0-bd830f810fba
∇²g(x) = -2g(x) * A - 2A * x * ∇g(x)'

# ╔═╡ 8fe85430-f2b5-11ea-3fe7-5d9a0db03932
g(xvect)

# ╔═╡ b372c25c-f2b5-11ea-1c55-e5045e0dc5b7
∇g(xvect)

# ╔═╡ b6e48b12-f2b5-11ea-3bf2-81e5b2a3dc98
∇g(xvect)' * dvect

# ╔═╡ b455e8ac-fe7f-11ea-1a8a-db85d11ccba7
grad_vect(g, xvect, dvect)

# ╔═╡ c2924514-fe7f-11ea-2b01-81e9e0e1c491
h = 1e-10

# ╔═╡ cd2821c4-fe7f-11ea-0c41-8feab49be07a
∇²g(xvect) * dvect

# ╔═╡ bebbd674-f2b5-11ea-19c6-4fe1257b0277
(∇g(xvect + h * dvect) - ∇g(xvect - h * dvect)) / 2h

# ╔═╡ d6c6d4ea-f2b7-11ea-0b41-d7f629cdcd2f
md"""
## Forward differentiation

Accumulation of the gradients along the *computational graph*.

![](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/forwarddiff.png)

Forward differentiation computes the gradient from the inputs to the outputs.

### Differentiation rules

**Sum rule**:

$$\frac{\partial (f(x)+g(x))}{\partial x} =  \frac{\partial f(x)}{\partial x} + \frac{\partial f(x)}{\partial x}$$

**Product rule**:

$$\frac{\partial (f(x)g(x))}{\partial x} =  f(x)\frac{\partial g(x)}{\partial x} + g(x)\frac{\partial f(x)}{\partial x}$$

**Chain rule**:

$$\frac{\partial (g(f(x))}{\partial x} =  \frac{\partial g(u)}{\partial u}\mid_{u=f(x)} \frac{\partial f(x)}{\partial x}$$

## Dual numbers

Forward differentiation can be viewed as evaluating function using *dual numbers*, which can be viewed as truncated Taylor series:

$$v + \dot{v}\epsilon\,,$$

where $v,\dot{v}\in\mathbb{R}$ and $\epsilon$ a nilpotent number, i.e. $\epsilon^2=0$. For example, we have

$$(v + \dot{v}\epsilon) + (u + \dot{u}\epsilon) = (v+u) + (\dot{v} +\dot{u})\epsilon$$


$$(v + \dot{v}\epsilon)(u + \dot{u}\epsilon) = (vu) + (v\dot{u} +\dot{v}u)\epsilon\,.$$


These dual numbers can be used as

$$f(v+\dot{v}\epsilon) = f(v) + f'(v)\dot{v}\epsilon\,.$$
"""

# ╔═╡ fe18341e-f2b7-11ea-0236-03b44eebd328
struct Dual{T}
    v::T
    vdot::T
end

# ╔═╡ 0aba93a6-f2b8-11ea-1461-d96878b7a5c1
md"Let's implement some basic rules showing linearity."

# ╔═╡ 014ec92c-f2b8-11ea-3986-8d74a8c2458d
begin
	Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.vdot + b.vdot)
	Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.vdot + b.v * a.vdot)
	Base.:+(c::Real, b::Dual) = Dual(c + b.v, b.vdot)
	Base.:*(v::Real, b::Dual) = Dual(v, 0.0) * b
	Base.:*(a::Dual, b::Real) = v * a
end

# ╔═╡ 3f5a58da-f2b8-11ea-3ab0-5db5e9199bb1
md"And some more advanced ones, based on differentiation."

# ╔═╡ 48e5da96-f2b8-11ea-0108-2104f2c8ac24
Base.:/(a::Dual, b::Dual) = Dual(a.v / b.v, (a.vdot * b.v - a.v * b.vdot) / b.v^2)

# ╔═╡ 263e18dc-fe70-11ea-03a0-8fd5e031b06f
md"**Assignment**: complete this code."

# ╔═╡ 2f18c362-f2b8-11ea-1beb-719ff5cd5f8d
Base.:sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.vdot)

# ╔═╡ 393f92a6-fe70-11ea-024e-ff30dcfc7103
Base.:cos(a::Dual) = Dual(cos(a.v), -sin(a.v) * a.vdot)

# ╔═╡ 329cb67e-f2b8-11ea-07b1-e13fd8ddbe42
Base.:exp(a::Dual) = Dual(exp(a.v), exp(a.v) * a.vdot)

# ╔═╡ 36a2ca10-f2b8-11ea-0fae-53b552c76115
Base.:log(a::Dual) = Dual(log(a.v), 1.0 / a.v * a.vdot)

# ╔═╡ 4ee45daa-f2b8-11ea-10ea-f929853879b1
f(Dual(a, 1.0))

# ╔═╡ 555c8d88-f2b8-11ea-34d8-a17ce87ccdc1
md"This directly works for vectors!"

# ╔═╡ 59eee760-f2b8-11ea-29f6-732292e2f95d
q(x) = 10.0 * x[1] * x[2] + x[1] * x[1] + sin(x[1]) / x[2]

# ╔═╡ 5ccd625e-f2b8-11ea-38c1-098e2f31ce53
q([1, 2])

# ╔═╡ 60b52f6e-f2b8-11ea-2afc-4be2d4ae3e19
q(Dual.([1, 2], [1, 0]))  # partial wrt x1

# ╔═╡ 63d5e0be-f2b8-11ea-16e8-d9528b4e8945
q(Dual.([1, 2], [0, 1]))  # partial wrt x2

# ╔═╡ 6971e7e6-f2b8-11ea-2a76-9dda6b1455b6
md"In practice, we prefer to use a package to do this."

# ╔═╡ 723fe4ca-f2b8-11ea-1440-a9e1cd213b66
ForwardDiff.derivative(f, a)

# ╔═╡ 76284520-f2b8-11ea-077e-43167cc7484c
ForwardDiff.gradient(g, xvect)

# ╔═╡ 85a5c518-f2b8-11ea-2e12-25d16a12e7c8
ForwardDiff.gradient(q, [1, 2])

# ╔═╡ 8a254852-f2b8-11ea-142d-71bb270a5c01
md"""
Forward differentiation:

- exact gradients!
- computational complexity scales with **number of inputs**
- used when you have more outputs than inputs
"""

# ╔═╡ 91d0e03e-f2b8-11ea-07e5-db056636ac21
md"""
# Reverse differentiation

Compute the gradient from the output toward the inputs using the chain rule.

![](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/reversediff.png)

Reverse differentiation:

- also exact!
- main workhorse for training artificial neural networks.
- efficient when more inputs than outputs (machine learning: thousands of parameters vs. one loss)
"""

# ╔═╡ a041fbf8-f2b8-11ea-28ba-0772497c649f
md"Constructing the derivative or gradient can be done by appending `'` to the function."

# ╔═╡ 9b5db398-f2b8-11ea-00bc-77de99792925
f'(a)  # that's it

# ╔═╡ bdce5964-f2b8-11ea-03d1-57705d295040
md"More verbose, but exactly the same:"

# ╔═╡ c6728784-f2b8-11ea-12b5-d7eb039d77ce
Zygote.gradient(f, a)  # returns a tuple, since you can differentiate wrt multiple arguments

# ╔═╡ e1057afe-fe6e-11ea-27fe-6def4d8e1f2d
md"This works for function with multiple inputs."

# ╔═╡ eb76e252-fe6e-11ea-0939-c51ff3c73827
Zygote.gradient((x1, x2) -> (x1^2 + x2^2 - 0.1x1*x2) / (x1 + 1.0),
					0.2, 0.3) 

# ╔═╡ d6d310ee-f2b8-11ea-24e1-3b405f071545
md"Fuctions with vectorial input."

# ╔═╡ d3ae2c14-f2b8-11ea-1ac2-fb5e0b4b8500
g'(xvect)

# ╔═╡ dbb1e68c-f2b8-11ea-1220-6f5715822e3d
md"Finding the Hessian:"

# ╔═╡ df557ed2-f2b8-11ea-22ee-e1c90a2655bf
Zygote.hessian(g, xvect)

# ╔═╡ ee52d2b8-f2b8-11ea-15c4-19e9f2ad10e2
md"""
## Artificial neural networks

Multi-layer perceptron.

![](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/ANN_example.png)

Forward differentiation.

![](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/Forwardprop.png)

Reverse differentation or backpropagation.

![](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.AutoDiff/Figures/Backprop.png)

Returns effect of changing layer output on the loss. Can be related directly to the parameters!
"""

# ╔═╡ 213fc38a-f380-11ea-135e-6f72924c19aa
md"""
## Differentiating complex objects

Automatic differentiation can be used beyond machine learning and optimization:

- [physical engines](https://arxiv.org/abs/1611.01652) to learn robot control
- differentiating [protein](https://github.com/lupoglaz/TorchProteinLibrary) [structures](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30076-6)
- Sinkhorn algorithm
- [dynamic programming](https://arxiv.org/abs/1802.03676)
- [differential equations](https://julialang.org/blog/2019/01/fluxdiffeq)

Everything is computed by some straightforward and differentiable functions!
"""

# ╔═╡ 79221eee-f2be-11ea-3e43-4596f5ce5694
md"""
## Application: inverse kinematics of a robot arm

As an illustration of using gradients, let us study a [simple robot arm](https://appliedgo.net/roboticarm/). This arm consists of three joints that can be moved in the 2D plane. These lengths of the three segments are, respectivly 2, 1 and 1.5 meter.

The forward kinetics (i.e., going from the angles of the joints to the position of the arms) can be computed by the following functions.
"""

# ╔═╡ fb91a9c4-f2c0-11ea-38c4-5b554446e997
pos_second_joint((θ₁, θ₂, θ₃)) = [2cos(θ₁), 2sin(θ₁)]

# ╔═╡ 07c45d78-f2c3-11ea-131e-b9f0b67ca3cd
pos_third_joint((θ₁, θ₂, θ₃)) = pos_second_joint((θ₁, θ₂, θ₃)) + [cos( θ₂), sin( θ₂)]

# ╔═╡ 33023d10-f2c1-11ea-3375-e9ac23172ab6
pos_hand((θ₁, θ₂, θ₃)) = pos_third_joint((θ₁, θ₂, θ₃)) .+ [1.5cos(θ₃), 1.5sin(θ₃)]

# ╔═╡ 31cdefd2-f381-11ea-3037-e9c023ebc1e9
md"Suppose that there is a teapot at a certain location we want to grasp with the arm."

# ╔═╡ 4def0626-f2c1-11ea-2f01-31a8c42b121c
position_teapot = [-1, 2.3]

# ╔═╡ 46c54108-f381-11ea-141f-31a696a7a8ef
md"Below is a plot of the arm and the teapot. Can you turn the angles to bring the hand to the position of the teapot?"

# ╔═╡ 8049560e-f383-11ea-1c8e-39c629d20982
@bind θ₁ Slider(0:0.01:2π, show_value=true)

# ╔═╡ c7b5a1aa-f383-11ea-2a84-8da9c9e60dbc
@bind θ₂ Slider(0:0.01:2π, show_value=true)

# ╔═╡ cfad1a8c-f383-11ea-3dad-214f3da9485d
@bind θ₃ Slider(0:0.01:2π, show_value=true)

# ╔═╡ 83ddbe6a-f381-11ea-2415-255cbfb2590b
md"First complete a function to determine the distance between the arm and the teapot."

# ╔═╡ 30e9891c-f2c5-11ea-022d-5dbf9c064e43
distance_to_teapot(θ) = (position_teapot .- pos_hand(θ)).^2 |> sum |> sqrt

# ╔═╡ 5812f84c-f2c1-11ea-0f6c-31b9fedc3704
begin
	psj = pos_second_joint((θ₁, θ₂, θ₃))
	ptj = pos_third_joint((θ₁, θ₂, θ₃))
	phand = pos_hand((θ₁, θ₂, θ₃))
	plot([0, psj[1], ptj[1], phand[1]], [0, psj[2],ptj[2], phand[2]], label="Arm")
	scatter!([0], [0], label="First joint (fixed)")
	scatter!([psj[1]], [psj[2]], label="Second joint")
	scatter!([ptj[1]], [ptj[2]], label="Second joint")
	scatter!([phand[1]], [phand[2]], label="Hand")
	scatter!([position_teapot[1]], [position_teapot[2]], m=:star, label="teapot")
	title!("Distance to teapot = $(distance_to_teapot((θ₁, θ₂, θ₃)))")
	xlims!(-5, 5)
	ylims!(-5, 5)
end

# ╔═╡ b07cb958-f381-11ea-3378-5517440464ca
md"Can you use gradients to position the arm?"

# ╔═╡ eaac2514-f8fe-11ea-2229-81f5b59f58fe
distance_to_teapot([θ₁, θ₂, θ₃])

# ╔═╡ fb044216-f381-11ea-2faf-fd53dad74950
distance_to_teapot'([θ₁, θ₂, θ₃])

# ╔═╡ 675b222c-f382-11ea-257a-afcb5a5afdd8
begin
	# initial value of θ
	θ = [0.0, 0.0, 0.0]
	for i in 1:100
		θ .-= 0.05distance_to_teapot'(θ)
	end
end

# ╔═╡ 818c1f9e-ff1b-11ea-097f-cba2f30af994
θ

# ╔═╡ 881dedd4-f385-11ea-342e-a3700e90ac66
md"""
## Exercise

Consider the *Wheeler's Ridge* function:

$$f(\mathbf{x}) = -\exp(-(x_1 x_2 - a)^2 -(x_2 -a)^2)\,,$$

at the point $\mathbf{x}_0=[1.5, 1]^T$. We set $a=1.5$.

Implement this function.
"""


# ╔═╡ c1e19e26-f385-11ea-17b6-6d2d2e5cb7c0
fwr(x) = missing

# ╔═╡ fe18a0a0-f8fe-11ea-1595-4d95256dabca
fwr([1,2]) isa Missing || contour(-2:0.1:2, -2:0.1:2, (x1, x2) -> fwr([x1, x2]))

# ╔═╡ df5f20d8-f385-11ea-104d-9b9e7b094eb7
md"""

**Assignments**

1. Compute the gradient by hand.
2. Find the gradient and Hessian at $\mathbf{x}_0$ by numerical differentiation.
3. Compute the gradient and Hessian at $\mathbf{x}_0$ using automatic differentiation.
4. (optional) Use the function `quiver!` to draw the gradient as a vector field on the contour plot.

"""

# ╔═╡ e4a088f0-f385-11ea-00da-b12e40e4357f
x₀ = [1.5, 1.0]  # btw written as x\_0<TAB>

# ╔═╡ 8f3fecac-f8fd-11ea-136d-1513f60ee3ed


# ╔═╡ f8863a18-f385-11ea-0965-0f074e58d9b2


# ╔═╡ 4c751e90-f387-11ea-2358-273423b85c00


# ╔═╡ 4d2c34ae-f387-11ea-1649-19c3a4520f74


# ╔═╡ 34d48ec0-f2b9-11ea-147e-d3554a9fcc04
md"""
# References

- Gunes et. al. (2015) *Automatic differentiation in machine learning: a survey*
- Kochenderfer, M. J. and Wheeler, T., '*Algorithms for Optimization*'. MIT Press (2019)
"""

# ╔═╡ Cell order:
# ╟─08cf2a80-f2b2-11ea-132d-2f26c934bbb2
# ╠═47f98a52-f2b2-11ea-0ac6-9b4b74b7b15d
# ╟─5154a672-f2b2-11ea-1280-3ba5dadff7f9
# ╟─f53fbf24-f2b2-11ea-2f59-d943884fee9d
# ╟─e6a55456-fe6f-11ea-328c-47fedb388194
# ╠═2be8656c-f2b3-11ea-2244-5740c807deff
# ╟─2e00ef04-f2b3-11ea-1960-8136321c71db
# ╟─f7cd052e-f2bb-11ea-239a-47966e96e909
# ╠═7244eba2-f2b3-11ea-045f-ffd208852eda
# ╟─401df7a4-f2b3-11ea-12a1-e9b46bcfcf1b
# ╠═5ac4989c-f2b3-11ea-3e8a-a79a4117e48c
# ╠═5db7350a-f2b3-11ea-293d-ffc572e0dbe8
# ╠═d5edd644-fe7c-11ea-2aa8-c772ea2f1bde
# ╠═64b28e4a-f2b3-11ea-0fc0-7d37b4794686
# ╠═684ff7f4-f2b3-11ea-14f7-7d3f329ecd79
# ╟─2ca60bca-f2b4-11ea-0fec-c50f4e306536
# ╟─7a6a18de-f2b3-11ea-1b3d-ef7a2b1b0434
# ╟─913234a2-f2b3-11ea-3501-e3bfc54f91fe
# ╟─0238579a-fe70-11ea-0968-0f11ae9557db
# ╠═eef066e0-f2b3-11ea-3cfa-5f35ebf13f76
# ╠═f2158710-f2b3-11ea-16c9-dded5e1747d4
# ╠═f54ee2b2-f2b3-11ea-24e8-f556ab9e3b85
# ╠═08a565e0-f2b4-11ea-2ab6-f5f3a1b3c22d
# ╠═1055a996-f2b4-11ea-14d2-6116cbbaa6ba
# ╠═13db037a-f2b4-11ea-2a38-bd7c1c60f7d9
# ╟─4f42234e-f2b4-11ea-2c86-93449b8dc60b
# ╠═6adbc7fe-f2b4-11ea-1861-b398d517edfa
# ╠═6cd1accc-f2b4-11ea-0077-e7685b77b1be
# ╟─77d4fd72-f2b4-11ea-0bd6-97fc2299ca2f
# ╠═773af248-f2b4-11ea-154a-cb67beebed3d
# ╟─8a5ee764-f2b4-11ea-23f3-27664d910c16
# ╠═8ded9128-f2b4-11ea-0f44-9321077205de
# ╟─c3a1abd8-f2b4-11ea-0509-a5296ae8b361
# ╠═c9f92e66-f2b4-11ea-322f-7778f149453d
# ╟─cfb07cc6-f2b4-11ea-08a7-dd0b13d15b5e
# ╠═d56b74f2-f2b4-11ea-12b4-5f54d4999b38
# ╟─d9f2d114-f2b4-11ea-32b0-43fa1a872a97
# ╠═de1fd700-f2b4-11ea-36dd-4bfba5ecc134
# ╠═df14ce72-f2b4-11ea-307b-8fac2889681b
# ╠═e584f976-f2b4-11ea-0245-67f6bcc4be0e
# ╟─eb68674c-f2b4-11ea-0cc2-f792d17a5005
# ╟─f4edad2c-f2b4-11ea-05fb-3d43c18e9d5c
# ╟─fd9ed266-f2b4-11ea-3c53-3bd2fe30a7f1
# ╟─4e6b2b04-f2b5-11ea-0c52-5f32438e8c68
# ╟─55ff7230-f2b5-11ea-2898-db5e21c764fd
# ╠═688a1f40-f2b5-11ea-3fa8-ebb735323ddd
# ╠═6d6da8ba-f2b5-11ea-1064-df27ecea7d13
# ╠═6fd08e42-f2b5-11ea-36e6-d7ff10412e2b
# ╠═9a87095e-f2b5-11ea-334e-57a637e20c43
# ╟─67e9fa88-ff03-11ea-1177-c34331398d6c
# ╠═75ae48fe-f2b5-11ea-0614-912091cbfb6a
# ╟─8a738272-f2b5-11ea-2de6-f9ae6b30a466
# ╠═7b95d73c-f2b5-11ea-1b29-858f7b6c99bf
# ╠═836ca94a-f2b5-11ea-07e0-bd830f810fba
# ╠═8fe85430-f2b5-11ea-3fe7-5d9a0db03932
# ╠═b372c25c-f2b5-11ea-1c55-e5045e0dc5b7
# ╠═b6e48b12-f2b5-11ea-3bf2-81e5b2a3dc98
# ╠═b455e8ac-fe7f-11ea-1a8a-db85d11ccba7
# ╠═c2924514-fe7f-11ea-2b01-81e9e0e1c491
# ╠═cd2821c4-fe7f-11ea-0c41-8feab49be07a
# ╠═bebbd674-f2b5-11ea-19c6-4fe1257b0277
# ╟─d6c6d4ea-f2b7-11ea-0b41-d7f629cdcd2f
# ╠═fe18341e-f2b7-11ea-0236-03b44eebd328
# ╟─0aba93a6-f2b8-11ea-1461-d96878b7a5c1
# ╠═014ec92c-f2b8-11ea-3986-8d74a8c2458d
# ╟─3f5a58da-f2b8-11ea-3ab0-5db5e9199bb1
# ╠═48e5da96-f2b8-11ea-0108-2104f2c8ac24
# ╟─263e18dc-fe70-11ea-03a0-8fd5e031b06f
# ╠═2f18c362-f2b8-11ea-1beb-719ff5cd5f8d
# ╠═393f92a6-fe70-11ea-024e-ff30dcfc7103
# ╠═329cb67e-f2b8-11ea-07b1-e13fd8ddbe42
# ╠═36a2ca10-f2b8-11ea-0fae-53b552c76115
# ╠═4ee45daa-f2b8-11ea-10ea-f929853879b1
# ╟─555c8d88-f2b8-11ea-34d8-a17ce87ccdc1
# ╠═59eee760-f2b8-11ea-29f6-732292e2f95d
# ╠═5ccd625e-f2b8-11ea-38c1-098e2f31ce53
# ╠═60b52f6e-f2b8-11ea-2afc-4be2d4ae3e19
# ╠═63d5e0be-f2b8-11ea-16e8-d9528b4e8945
# ╟─6971e7e6-f2b8-11ea-2a76-9dda6b1455b6
# ╠═6fe82e5a-f2b8-11ea-101d-d1ee5f9bd5b1
# ╠═723fe4ca-f2b8-11ea-1440-a9e1cd213b66
# ╠═76284520-f2b8-11ea-077e-43167cc7484c
# ╠═85a5c518-f2b8-11ea-2e12-25d16a12e7c8
# ╟─8a254852-f2b8-11ea-142d-71bb270a5c01
# ╟─91d0e03e-f2b8-11ea-07e5-db056636ac21
# ╠═963c4910-f2b8-11ea-12bb-bb42539b1c93
# ╟─a041fbf8-f2b8-11ea-28ba-0772497c649f
# ╠═9b5db398-f2b8-11ea-00bc-77de99792925
# ╟─bdce5964-f2b8-11ea-03d1-57705d295040
# ╠═c6728784-f2b8-11ea-12b5-d7eb039d77ce
# ╟─e1057afe-fe6e-11ea-27fe-6def4d8e1f2d
# ╠═eb76e252-fe6e-11ea-0939-c51ff3c73827
# ╟─d6d310ee-f2b8-11ea-24e1-3b405f071545
# ╠═d3ae2c14-f2b8-11ea-1ac2-fb5e0b4b8500
# ╟─dbb1e68c-f2b8-11ea-1220-6f5715822e3d
# ╠═df557ed2-f2b8-11ea-22ee-e1c90a2655bf
# ╟─ee52d2b8-f2b8-11ea-15c4-19e9f2ad10e2
# ╟─213fc38a-f380-11ea-135e-6f72924c19aa
# ╟─79221eee-f2be-11ea-3e43-4596f5ce5694
# ╠═fb91a9c4-f2c0-11ea-38c4-5b554446e997
# ╠═07c45d78-f2c3-11ea-131e-b9f0b67ca3cd
# ╠═33023d10-f2c1-11ea-3375-e9ac23172ab6
# ╟─31cdefd2-f381-11ea-3037-e9c023ebc1e9
# ╠═4def0626-f2c1-11ea-2f01-31a8c42b121c
# ╟─46c54108-f381-11ea-141f-31a696a7a8ef
# ╟─5812f84c-f2c1-11ea-0f6c-31b9fedc3704
# ╠═8049560e-f383-11ea-1c8e-39c629d20982
# ╠═c7b5a1aa-f383-11ea-2a84-8da9c9e60dbc
# ╠═cfad1a8c-f383-11ea-3dad-214f3da9485d
# ╟─83ddbe6a-f381-11ea-2415-255cbfb2590b
# ╠═30e9891c-f2c5-11ea-022d-5dbf9c064e43
# ╟─b07cb958-f381-11ea-3378-5517440464ca
# ╠═eaac2514-f8fe-11ea-2229-81f5b59f58fe
# ╠═fb044216-f381-11ea-2faf-fd53dad74950
# ╠═675b222c-f382-11ea-257a-afcb5a5afdd8
# ╠═818c1f9e-ff1b-11ea-097f-cba2f30af994
# ╟─881dedd4-f385-11ea-342e-a3700e90ac66
# ╠═c1e19e26-f385-11ea-17b6-6d2d2e5cb7c0
# ╟─fe18a0a0-f8fe-11ea-1595-4d95256dabca
# ╟─df5f20d8-f385-11ea-104d-9b9e7b094eb7
# ╠═e4a088f0-f385-11ea-00da-b12e40e4357f
# ╠═8f3fecac-f8fd-11ea-136d-1513f60ee3ed
# ╠═f8863a18-f385-11ea-0965-0f074e58d9b2
# ╠═4c751e90-f387-11ea-2358-273423b85c00
# ╠═4d2c34ae-f387-11ea-1649-19c3a4520f74
# ╟─34d48ec0-f2b9-11ea-147e-d3554a9fcc04
