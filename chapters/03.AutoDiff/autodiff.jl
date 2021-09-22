### A Pluto.jl notebook ###
# v0.16.0

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
using Symbolics

# ╔═╡ 6fe82e5a-f2b8-11ea-101d-d1ee5f9bd5b1
using ForwardDiff

# ╔═╡ 963c4910-f2b8-11ea-12bb-bb42539b1c93
using Zygote

# ╔═╡ 08cf2a80-f2b2-11ea-132d-2f26c934bbb2
md"""
# Numeric and automatic differentiation

STMO

![](https://github.com/MichielStock/STMO/blob/master/chapters/03.AutoDiff/Figures/logo.png?raw=true)

**Michiel Stock**

"""

# ╔═╡ b72a9871-3773-4a06-9a2f-09ce5a7d8496
md"## Motivation

Up to now, we confidently assumed that we would always be able to compute the derivative or gradient of any function. Despite differentiation being a relatively easy operation, it is frequenty not feasible (or desirable) to compute this by hand. *Numerical differentiation* can provide approximations of th derivate or gradient at a particular point. *Automatic differentiation* directly manipulates the computational graph to generate a function that computes the (exact) derivate. Such methods have advanced greatly in the last years and it is no exageration that their easy use in popular software libraries such as TenserFlow and PyTorch are a cornerstone of deep learning and other machine learning and scientific computing fields.
"

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
md"**Assignment**: implement this function. (or another one)"

# ╔═╡ 2be8656c-f2b3-11ea-2244-5740c807deff
f(x) = log(x) + sin(x) / x

# ╔═╡ f7cd052e-f2bb-11ea-239a-47966e96e909
md"We will evaluate this function in `a`=$2."

# ╔═╡ 7244eba2-f2b3-11ea-045f-ffd208852eda
a = 2.0

# ╔═╡ 2e00ef04-f2b3-11ea-1960-8136321c71db
ismissing(f(a)) || plot(f, 1, 5, label="f(x)")

# ╔═╡ 401df7a4-f2b3-11ea-12a1-e9b46bcfcf1b
md"""
## Symbolic differentiation

Computing derivatives, as you have seen in basic calculus courses.

By hand or automatically:
- Maple
- Sympy (Python)
- Mathematica
- Maxima
- Symbolics (Julia)

Differentiation is *easy* compared to *integration* or *sampling*.

Advantages:
- exact derivatives!
- gives the formula for different evaluations.
    - insight in the system (i.e. variable are independent can be learned from second-order partial derivatives)
    - in some cases, closed-form solution extrema by solving $\frac{\text{d}f(x)}{\text{d}x}=0$
- no hyperparameters or tweaking: just works!

Disadvantages:
- some software not flexible enough (gradients, arrays, for-loops,...)
- sometimes explosion of terms: *expression swell*
- not always numerically optimal!
"""

# ╔═╡ 5db7350a-f2b3-11ea-293d-ffc572e0dbe8
@variables x  # define variable

# ╔═╡ d5edd644-fe7c-11ea-2aa8-c772ea2f1bde
f(x)

# ╔═╡ 76ba97fe-b7c4-467b-85ae-9067ebcc8b14
Dx = Differential(x)  # differential operator

# ╔═╡ 64b28e4a-f2b3-11ea-0fc0-7d37b4794686
Dx(f(x))

# ╔═╡ 8a4a8499-af01-46ea-8196-fb2d718d0d17
df_sym = expand_derivatives(Dx(f(x)))  # this expands the derviatve operator

# ╔═╡ 92b65737-9ad7-422e-955c-e9adacd2032b
md"We can build this expression in a function:"

# ╔═╡ a38e2cce-851c-4065-8947-1926fa9ca5b4
df = build_function(df_sym, x) |> eval  #builds an expression and turns it into a function

# ╔═╡ 684ff7f4-f2b3-11ea-14f7-7d3f329ecd79
df(a)

# ╔═╡ 2ca60bca-f2b4-11ea-0fec-c50f4e306536
true_diff = df(a);  # save for checking

# ╔═╡ 7a6a18de-f2b3-11ea-1b3d-ef7a2b1b0434
if !ismissing(f(a))
plot(f, 1, 10, label="\$f(x)\$", xlabel="\$x\$", lw=2, color=:green)
plot!(df, 1, 10, label="\$f'(x)\$", lw=2, color=:orange)
end

# ╔═╡ 67e7ea4d-0b76-4247-b22d-e20c8280f046
md"You can play with this dynamically:"

# ╔═╡ 45190ad2-d358-455a-89a1-4f9c3b3880fa
@bind n Slider(-10:10, show_value=true, default=2)

# ╔═╡ 732ad756-1707-48e7-9bfb-10707e64b118
n

# ╔═╡ c8a5c6f8-4ca1-4c7e-9cc1-d2e5c9e9da59
symexpr = (x + 2)^n / (x+2)

# ╔═╡ c67f8b3f-0561-4f47-ade8-3c79d7920ae1
md"Derivative:"

# ╔═╡ 997ec642-a865-4c97-9aa1-d4d9df9e1dd0
expand_derivatives(Dx(symexpr)) |> simplify

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
diff_fordiff(f, x; h=1e-10) = missing

# ╔═╡ f2158710-f2b3-11ea-16c9-dded5e1747d4
diff_centrdiff(f, x; h=1e-10) = missing

# ╔═╡ f54ee2b2-f2b3-11ea-24e8-f556ab9e3b85
diff_complstep(f, x; h=1e-10) = missing

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

# ╔═╡ 3f2a4cc2-26fa-4c3e-bfb7-f3ef019b6d78
Base.show(io::IO, a::Dual) = print(io, "$(a.v) ± $(a.vdot)")  # nice printing

# ╔═╡ 9e22156e-0880-11eb-24a8-3d0a41262c36
ϵ = Dual(0.0, 1.0)

# ╔═╡ ab39b554-0880-11eb-05de-71949f71b440
Dual(2.0, 3.0)

# ╔═╡ 0aba93a6-f2b8-11ea-1461-d96878b7a5c1
md"Let's implement some basic rules showing linearity."

# ╔═╡ 4a6270a3-a271-40da-84d0-28247f287c22
2.0 + 3.0ϵ  # now this works!

# ╔═╡ 3f5a58da-f2b8-11ea-3ab0-5db5e9199bb1
md"And some more advanced ones, based on differentiation."

# ╔═╡ 48e5da96-f2b8-11ea-0108-2104f2c8ac24
Base.:/(a::Dual, b::Dual) = Dual(a.v / b.v, (a.vdot * b.v - a.v * b.vdot) / b.v^2)

# ╔═╡ 263e18dc-fe70-11ea-03a0-8fd5e031b06f
md"**Assignment**: complete this code."

# ╔═╡ 2f18c362-f2b8-11ea-1beb-719ff5cd5f8d
Base.:sin(a::Dual) = missing

# ╔═╡ 393f92a6-fe70-11ea-024e-ff30dcfc7103
Base.:cos(a::Dual) = missing

# ╔═╡ 329cb67e-f2b8-11ea-07b1-e13fd8ddbe42
Base.:exp(a::Dual) = missing

# ╔═╡ 36a2ca10-f2b8-11ea-0fae-53b552c76115
Base.:log(a::Dual) = missing

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

# ╔═╡ 059679c0-d0ff-453b-b897-b889bbea326d
md"""
### Example: differentiable sequence alignment

One can differentiate the Needleman-Wunsch algorithm for sequence alignment. This yields insight in how the alignment score would change if one changes the alignment parameters.
"""

# ╔═╡ a20f0158-f32b-45e8-8bd5-6f693eca1860
abstract type Regularizer end

# ╔═╡ 23b0cc81-b551-446b-991f-54bd19ac63cd
struct NegEntropy{T<:Number} <: Regularizer
    γ::T
end

# ╔═╡ a9961630-08be-48f8-b1cc-4a9d67dfc519
function max_argmax(Ω::NegEntropy, x)
    γ = Ω.γ
    m = maximum(x)
    # stable log-sum-exp computation
    lse = m + γ * log(sum(exp, (x .- m) ./ γ))
    # stable softmax
    q = exp.((x .- lse) ./ γ)
    return lse, q  # smooth max and grad
end

# ╔═╡ e4d2350e-147b-4c3c-a9ff-6de66a54c48a
function ∇needleman_wunsch(Ω::Regularizer, θ, (cˢ, cᵗ))
    n, m = size(θ)
    D = zeros(n+1, m+1)        # initialize DP matrix
    D[2:n+1,1] .= -cumsum(cˢ)  # cost of starting with gaps in s
    D[1,2:m+1] .= -cumsum(cᵗ)  # cost of starting with gaps in t
    E = zeros(n+2, m+2)        # matrix for the gradient
    E[n+2,m+2] = 1.0
    Q = zeros(n+2, m+2, 3)     # matrix for backtracking
    Q[n+2,m+2,2] = 1.0  
    # forward pass, performing dynamic programming
    for i in 1:n, j in 1:m
        v, q = max_argmax(Ω, (D[i+1,j] - cˢ[i],   # gap in s
                              D[i,j] + θ[i,j],    # match
                              D[i,j+1] - cᵗ[j]))  # gap in t
        D[i+1,j+1] = v      # store smooth max
        Q[i+1,j+1,:] .= q   # store directions
    end
    v = D[n+1,m+1]        # get alignment score
    # backtracking through the directions to compute the gradient
    for i in n:-1:1, j in m:-1:1
        E[i+1,j+1] = Q[i+1,j+2,1] * E[i+1,j+2] +  
                     Q[i+2,j+2,2] * E[i+2,j+2] +
                     Q[i+2,j+1,3] * E[i+2,j+1]
    end
    return v, E[2:n+1,2:m+1]  # value and gradient
end

# ╔═╡ 501a0db3-538a-43f3-a769-158c824bca31
md"For example, consider two sequences and the Hamming distance."

# ╔═╡ ca64df96-082a-40dd-b82c-64016257fcab
s, t = "banana", "ananas"

# ╔═╡ ba43bc8b-27d6-490d-971e-7e440b991cfb
θ = [sᵢ==tᵢ for sᵢ in s, tᵢ in t]

# ╔═╡ 2d7dbec3-82eb-41f8-a959-b458e136d18c
heatmap(θ, flipy=true, yticks=(1:length(s), s), xticks=(1:length(t), t), title="theta")

# ╔═╡ 378500c1-9f19-4111-82cf-5a57c30d66ef
cˢ, cᵗ = ones(length(s)), ones(length(t))

# ╔═╡ 8586855b-2ad7-4fe4-af1a-2a6f1dd052b3
@bind logγ Slider(-3:0.2:4, default=0)

# ╔═╡ 8877c563-d10a-41e4-be8b-546d27623b67
γ = exp(logγ)

# ╔═╡ a73e4d59-34f4-412f-afa4-f3b66244b35e
γ

# ╔═╡ 6707fd27-78f6-4663-b95d-a016d2c44c3b
# alignment value and score
v, E = ∇needleman_wunsch(NegEntropy(γ), θ, (cˢ, cᵗ))

# ╔═╡ 014ec92c-f2b8-11ea-3986-8d74a8c2458d
begin
	Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.vdot + b.vdot)
	Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.vdot + b.v * a.vdot)
	Base.:+(c::Real, b::Dual) = Dual(c + b.v, b.vdot)
	Base.:*(v::Real, b::Dual) = Dual(v, 0.0) * b
	Base.:*(a::Dual, b::Real) = v * a
end

# ╔═╡ 18e151c4-0b0e-485b-81d2-f18a2dd4592d
heatmap(E, flipy=true, yticks=(1:length(s), s), xticks=(1:length(t), t), title="gradient of NW", color=:speed)

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
	θs = [0.0, 0.0, 0.0]
	for i in 1:100
		θs .-= 0.05distance_to_teapot'(θs)
	end
end

# ╔═╡ 818c1f9e-ff1b-11ea-097f-cba2f30af994
θs

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
2. Compute the gradient using symbolic differentiation.
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

# ╔═╡ 0a576c65-3685-4987-93f1-980ace78d6f4
module Solution

	export diff_fordiff, diff_centrdiff, diff_complstep
	
	diff_fordiff(f, x; h=1e-10) = (f(x + h) - f(x)) / h
	diff_centrdiff(f, x; h=1e-10) = (f(x + h) - f(x - h)) / 2h
	diff_complstep(f, x; h=1e-10) = imag(f(x + im * h)) / h

	
end

# ╔═╡ 8eb9ed90-981b-4094-baec-a521614fb3a1
md"solution dual numbers:
```julia
	Base.:sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.vdot)
	Base.:cos(a::Dual) = Dual(cos(a.v), -sin(a.v) * a.vdot)
	Base.:exp(a::Dual) = Dual(exp(a.v), exp(a.v) * a.vdot)
	Base.:log(a::Dual) = Dual(log(a.v), 1.0 / a.v * a.vdot)
```"

# ╔═╡ fe4dfa5c-ea26-4f7c-aa68-7b3ec5f45bdc
begin
	myblue = "#304da5"
	mygreen = "#2a9d8f"
	myyellow = "#e9c46a"
	myorange = "#f4a261"
	myred = "#e76f51"
	myblack = "#50514F"

	mycolors = [myblue, myred, mygreen, myorange, myyellow]
end;

# ╔═╡ fd9ed266-f2b4-11ea-3c53-3bd2fe30a7f1
begin
	fexamp(x) = 64x*(1-x)*(1-2x)^2*(1-8x+8x^2)^2
	#dfexamp = diff(fexamp(x), x)
	dfexamp = build_function(expand_derivatives(Dx(fexamp(x))), x) |> eval
	error(diff, h; x=1.0) = max(abs(Float64(dfexamp(x)) - diff(fexamp, x, h=h)), 1e-50)
	stepsizes = map(t->10.0^t, -20:0.1:-1);
	plot(stepsizes, error.(Solution.diff_fordiff, stepsizes), label="forward difference",
    xscale=:log10, yscale=:log10, lw=2, legend=:bottomright, color=myblue)
	plot!(stepsizes, error.(Solution.diff_centrdiff, stepsizes), label="central difference", 		lw=2, color=myred)
	plot!(stepsizes, error.(Solution.diff_complstep, stepsizes), label="complex step", lw=2,
            color=myyellow)
	xlabel!("\$h\$")
	ylabel!("absolute error")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ForwardDiff = "~0.10.19"
Plots = "~1.22.1"
PlutoUI = "~0.7.9"
Symbolics = "~3.3.1"
Zygote = "~0.6.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bijections]]
git-tree-sha1 = "705e7822597b432ebe152baa844b49f8026df090"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.3"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "d88340ab502af66cfffc821e70ae72f7dbdce645"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.11.5"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4ce9393e871aca86cc457d9f66976c3da6902ea7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "627844a59d3970db8082b778e53f86741d17aaad"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.7"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "05b68e727a192783be0b34bd8fee8f678505c0bf"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.3.20"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "caf289224e622f518c9dbfe832cdafa17d7c80a6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.4"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LabelledArrays]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "bdde43e002847c34c206735b1cf860bc3abd35e7"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.6.4"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "45c9940cec79dedcdccc73cc6dd09ea8b8ab142c"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.3.18"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "4c2637482176b1c2fb99af4d83cb2ff0328fc33c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "00bede2eb099dcc1ddc3f9ec02180c326b420ee2"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.17.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "ff686e0c79dbe91767f4c1e44257621a5455b1c6"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.18.7"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "b680da4a404767b41044b660665b3d7f56016cd5"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.15.5"

[[Symbolics]]
deps = ["ConstructionBase", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TreeViews"]
git-tree-sha1 = "0e4d48d9c416563e3b067759e3cd72066d4f491b"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "3.3.1"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TermInterface]]
git-tree-sha1 = "02a620218eaaa1c1914d228d0e75da122224a502"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.1.8"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "209a8326c4f955e2442c07b56029e88bb48299c7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.12"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "ffbf36ba9cd8476347486a013c93590b910a4855"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.21"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─08cf2a80-f2b2-11ea-132d-2f26c934bbb2
# ╟─b72a9871-3773-4a06-9a2f-09ce5a7d8496
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
# ╠═76ba97fe-b7c4-467b-85ae-9067ebcc8b14
# ╠═64b28e4a-f2b3-11ea-0fc0-7d37b4794686
# ╠═8a4a8499-af01-46ea-8196-fb2d718d0d17
# ╟─92b65737-9ad7-422e-955c-e9adacd2032b
# ╠═a38e2cce-851c-4065-8947-1926fa9ca5b4
# ╠═684ff7f4-f2b3-11ea-14f7-7d3f329ecd79
# ╟─2ca60bca-f2b4-11ea-0fec-c50f4e306536
# ╟─7a6a18de-f2b3-11ea-1b3d-ef7a2b1b0434
# ╟─67e7ea4d-0b76-4247-b22d-e20c8280f046
# ╟─45190ad2-d358-455a-89a1-4f9c3b3880fa
# ╠═732ad756-1707-48e7-9bfb-10707e64b118
# ╠═c8a5c6f8-4ca1-4c7e-9cc1-d2e5c9e9da59
# ╟─c67f8b3f-0561-4f47-ade8-3c79d7920ae1
# ╟─997ec642-a865-4c97-9aa1-d4d9df9e1dd0
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
# ╠═3f2a4cc2-26fa-4c3e-bfb7-f3ef019b6d78
# ╠═9e22156e-0880-11eb-24a8-3d0a41262c36
# ╠═ab39b554-0880-11eb-05de-71949f71b440
# ╟─0aba93a6-f2b8-11ea-1461-d96878b7a5c1
# ╠═014ec92c-f2b8-11ea-3986-8d74a8c2458d
# ╠═4a6270a3-a271-40da-84d0-28247f287c22
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
# ╟─059679c0-d0ff-453b-b897-b889bbea326d
# ╠═a20f0158-f32b-45e8-8bd5-6f693eca1860
# ╠═23b0cc81-b551-446b-991f-54bd19ac63cd
# ╠═a9961630-08be-48f8-b1cc-4a9d67dfc519
# ╠═e4d2350e-147b-4c3c-a9ff-6de66a54c48a
# ╠═501a0db3-538a-43f3-a769-158c824bca31
# ╠═ca64df96-082a-40dd-b82c-64016257fcab
# ╠═ba43bc8b-27d6-490d-971e-7e440b991cfb
# ╟─2d7dbec3-82eb-41f8-a959-b458e136d18c
# ╠═378500c1-9f19-4111-82cf-5a57c30d66ef
# ╟─8586855b-2ad7-4fe4-af1a-2a6f1dd052b3
# ╠═8877c563-d10a-41e4-be8b-546d27623b67
# ╠═a73e4d59-34f4-412f-afa4-f3b66244b35e
# ╠═6707fd27-78f6-4663-b95d-a016d2c44c3b
# ╟─18e151c4-0b0e-485b-81d2-f18a2dd4592d
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
# ╠═0a576c65-3685-4987-93f1-980ace78d6f4
# ╟─8eb9ed90-981b-4094-baec-a521614fb3a1
# ╟─fe4dfa5c-ea26-4f7c-aa68-7b3ec5f45bdc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
