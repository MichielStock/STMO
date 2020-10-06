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

# ╔═╡ 49e8e6f6-0483-11eb-0a74-d94aa536c9ab
using LinearAlgebra, Plots, Zygote, PlutoUI

# ╔═╡ fccca5fc-0597-11eb-1033-a5cc3b71d854
md"""
# Constrained convex optimization

**Michiel Stock**
STMO

## Equality constraints

Consider the following optimization problem:

$$\min_{\mathbf{x}} f(\mathbf{x})$$
$$\text{subject to } g(\mathbf{x})=0\,.$$

For every point $\mathbf{x}$ on the surface $g(\mathbf{x})=0$, the gradient $\nabla g(\mathbf{x})$ is normal to this surface. This can be shown by considering a point $\mathbf{x}+\boldsymbol{\epsilon}$, also on the surface. If we make a Taylor expansion around $\mathbf{x}$, we have

$$g(\mathbf{x}+\boldsymbol{\epsilon})\approx g(\mathbf{x}) + \boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})\,.$$

Given that both $\mathbf{x}$ and $\mathbf{x}+\boldsymbol{\epsilon}$ lie on the surface it follows that $g(\mathbf{x}+\boldsymbol{\epsilon})= g(\mathbf{x})$. In the limit that $||\boldsymbol{\epsilon}||\rightarrow 0$ we have that $\boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})=0$. Because $\boldsymbol{\epsilon}$ is parallel to the surface $g(\mathbf{x})$, it follows that $\nabla g(\mathbf{x})$ is normal to the surface.

We seek a point $\mathbf{x}^\star$ on the surface such that $f(\mathbf{x})$ is minimized. For such a point, it should hold that the gradient w.r.t. $f$ should be parallel to $\nabla g$. Otherwise, it would be possible to give a small 'nudge' to $\mathbf{x}^\star$ in the direction of $\nabla f$ to decrease the function value, which would indicate that $\mathbf{x}^\star$ is not a minimizer. This figures below illustrate this point.

$$\nabla f(\mathbf{x}^\star) + \nu \nabla g (\mathbf{x}^\star)=0\,,$$
with $\nu\neq 0$ called the *Lagrange multiplier*. The constrained minimization problem can also be represented by a *Lagrangian*:
$$L(\mathbf{x}, \nu) 	\equiv f(\mathbf{x}) + \nu g(\mathbf{x})\,.$$
The constrained stationary condition is obtained by setting $\nabla_\mathbf{x} L(\mathbf{x}, \nu) =0$, the condition $\partial  L(\mathbf{x}, \nu)/\partial \nu=0$ leads to the constraint equation $g(\mathbf{x})=0$.


"""

# ╔═╡ 5e4eabba-0483-11eb-09a4-31ae6c582d3e
f((x1, x2)) = 2x1^2 +1.4x1*x2 + x2^2 - 0.3x1 + x1 

# ╔═╡ 470048ec-0598-11eb-0d48-67357ed1f12c
md"In addition to a $g(\mathbf{x})$, we also implement a parametric version where $g(\mathbf{x})=0$ for plotting purposes."

# ╔═╡ 5503c842-0591-11eb-0104-359b9ebd5e98
md"Show the countour of $f(\mathbf{x})$:"

# ╔═╡ 47067fc8-058c-11eb-224c-6fe2e470107f
@bind show_f_contour CheckBox()

# ╔═╡ c369c65e-0598-11eb-0a90-dd4db070c99b
show_f_contour

# ╔═╡ 60328136-0591-11eb-1929-b7515a8c03d9
md"Show the countour of $g(\mathbf{x})$:"

# ╔═╡ d10b14c2-058c-11eb-3a43-7bca5d042969
@bind show_g_contour CheckBox()

# ╔═╡ ba835a96-0598-11eb-1d54-9365e3ea337f
show_g_contour

# ╔═╡ 681a031a-0591-11eb-2fec-8d94eca758f8
md"Show the function $g(\mathbf{x})=0$:"

# ╔═╡ 13467b42-058d-11eb-3023-b9d3f145335b
@bind show_g0_constraint CheckBox()

# ╔═╡ ca227ac2-0598-11eb-3a39-f1a90769ac52
show_g0_constraint

# ╔═╡ d8276f26-0598-11eb-1626-95f87cf6de22
md"Show the gradients:"

# ╔═╡ 60c41848-0592-11eb-39ef-7103ae834990
@bind show_gradients CheckBox()

# ╔═╡ d16723ca-0598-11eb-02f4-f1c5d45095a5
show_gradients

# ╔═╡ 95dc760a-058e-11eb-2149-898b2776079f
@bind t Slider(0:0.01:2π, default=4.3)

# ╔═╡ 41d8c8d0-0591-11eb-2cd7-3b53f91ef589
t

# ╔═╡ aef30ee8-0597-11eb-2203-ed0f9b14f0e4
t

# ╔═╡ 00a009a2-0599-11eb-341f-3364198e1753
md"We plot the objective value on the equality constraint."

# ╔═╡ 13b4ad4a-0599-11eb-0043-6bfc0eb23389
md"At the minimizer, the gradients are parallel."

# ╔═╡ 7e77e628-0707-11eb-1b2f-27b5c5819260
# compute angle between two vectors
angle(u, v) = acos(dot(u, v) / (norm(u) * norm(v)))

# ╔═╡ 358f21ca-0599-11eb-3bfb-0bb12a1465bb
md"""
## Inequality constraints

The same argument can be made for inequality constraints, i.e. solving

$$\min_{\mathbf{x}} f(\mathbf{x})$$
$$\text{subject to } g(\mathbf{x})\leq0\,.$$

Here, two situations can arise:

- **Inactive constraint**: the minimizer of $f$ lies in the region where $g(\mathbf{x}) < 0$. This corresponds to a Lagrange multiplier $\nu=0$. Note that the solution would be the same if the constraint was not present.
- **Active constraint**: the minimizer of $f$ lies in the region where $g(\mathbf{x}) > 0$. The solution of the constrained problem will lie on the bound where $g(\mathbf{x})=0$, similar to the equality-constrained problem and corresponds to a Lagrange multiplier $\nu>0$.

For both cases, the product $\nu g(\mathbf{x})=0$, the solution should thus satisfy the following conditions:
$$g(\mathbf{x}) \leq 0$$
$$\nu \geq 0$$
$$\nu g(\mathbf{x})=0\,.$$
These are called the *Karush-Kuhn-Tucker* conditions.

It is relatively straightforward to extend this framework towards multiple constraints (equality and inequality) by using several Lagrange multipliers.
"""


# ╔═╡ 2d7ae208-0599-11eb-1a91-277b764ad45e
md"Change the location of the constraint"

# ╔═╡ c98b4f9a-0592-11eb-2034-5f05e792cbd2
@bind R Slider(10:0.5:25, default=16)

# ╔═╡ bd7b55ae-0599-11eb-1056-ed3682fe8160
g((x1, x2)) = (x1-10)^2 + (x2-17)^2 - R^2

# ╔═╡ c10838a2-0599-11eb-32e3-1579a926f9b7
g0(t) = [R*cos(t) + 10, R*sin(t) + 17]

# ╔═╡ d2c4ffe2-0485-11eb-0e20-dd52446fa3a6
x = g0(t)

# ╔═╡ 6e89822c-070b-11eb-06f3-8b8e20aaa57d
f(x)

# ╔═╡ e944567a-0596-11eb-2cf1-75f196bab4d1
f'(x) / norm(f'(x))

# ╔═╡ 02673cec-0597-11eb-2db4-e9f7d9047ac0
g'(x) / norm(g'(x))

# ╔═╡ c04a0ffe-0707-11eb-2a74-b3b258b4aec6
angle(f'(x), g'(x))  # close to 0 or pi if stationary point

# ╔═╡ f47ce686-0485-11eb-2a33-4bd0b150b18b
g(x)

# ╔═╡ b0cf9a40-058f-11eb-13dc-b7c35328ef2a
begin
	plot(t -> f(g0(t)), 0:0.01:2π, label="f(x) where g(x)=0")
	xlabel!("t")
	ylabel!("objective")
	scatter!([t], [f(g0(t))], color=:purple, label="x")
end

# ╔═╡ be35a06c-0596-11eb-2988-61a79c7e6a87
R

# ╔═╡ a08e7230-0591-11eb-3956-fb6c2d756494
fconstr(x) = g(x) ≤ 0.0 ? f(x) : NaN

# ╔═╡ 81629936-0593-11eb-3fca-afa68e2ace3d
@bind t_approx Slider(0.1:0.05:10.0, default=1.0)

# ╔═╡ a1402cbc-0593-11eb-33b4-09a0c654eefd
t_approx

# ╔═╡ cc4209ee-0593-11eb-3762-e76539e2a455
Î₋(u) = u < 0.0 ? -(1/t_approx)* log(-u) : Inf

# ╔═╡ a398740e-0593-11eb-2f81-c92861e080b8
begin
	plot(-3:0.001:2, Î₋, label="logarithmic barier (t=$t_approx)")
	plot!([-3, 0, 0], [0, 0, 5], ls=:dash, label="indicator function")
	ylims!(-2, 5)
	xlabel!("u")
end

# ╔═╡ dce25666-0594-11eb-24f9-85eba4949d1e
fsoft(x) = g(x) ≤ 0.0 ? t_approx * f(x) + Î₋(g(x)) : NaN

# ╔═╡ 701efa2a-0596-11eb-2592-39eced2fb875
@bind show_soft_grads CheckBox()

# ╔═╡ 7c5d381a-0596-11eb-1dfe-0b7074093cd6
show_soft_grads

# ╔═╡ 7c934678-0599-11eb-1e6b-1f2c9003509a
md"## Utilities"

# ╔═╡ 865bbb8e-0483-11eb-0622-b9a8a6ba5d4c
x1min, x1max = -8.0, 15.0

# ╔═╡ 9736f388-0483-11eb-069a-0d5ed87efb73
x2min, x2max = -8.0, 12.0

# ╔═╡ f39f0106-0483-11eb-3617-5fb04388489e
grads(f, x) = 0.1f'(x) |> df -> Tuple([e] for e in df)

# ╔═╡ 828cbe72-0483-11eb-33b2-7f4f37db889f
begin
	plot(colorbar=false)
	show_f_contour && contourf!(x1min:0.1:x1max, x2min:0.1:x2max, (x1, x2) -> f((x1, x2)), label="f(x)", colorbar=true, color=:Blues, aspect=:equal)
	show_g_contour && contour!(x1min:0.1:x1max, x2min:0.1:x2max, (x1, x2) -> g((x1, x2)), color=:Reds)
	show_g0_constraint && plot!([g0(t)[1] for t in 0:0.01:2π], [g0(t)[2] for t in 0:0.01:2π], lw=2, color=:red, label="g(x)=0")
	show_g0_constraint && scatter!([x[1]], [x[2]], label="x", color=:purple)
	scatter!([0], [0], label="x* (no constraint)", color=:green)
	show_gradients && quiver!([x[1]], [x[2]], quiver=grads(f, x), color=:blue)
	show_gradients && quiver!([x[1]], [x[2]], quiver=grads(g, x), color=:orange)
	xlims!(x1min, x1max)
	ylims!(x2min, x2max)
	xlabel!("x1")
	ylabel!("x2")
end

# ╔═╡ c2640938-0591-11eb-058c-5949d1e316d3
begin
	contourf(x1min:0.01:x1max, x2min:0.01:x2max, (x1, x2) -> fconstr((x1, x2)), 			color=:Blues)
	show_g_contour && contour!(x1min:0.1:x1max, x2min:0.1:x2max, (x1, x2) -> g((x1, x2)), color=:Reds)
	contour!(x1min:0.1:x1max, x2min:0.1:x2max, (x1, x2) -> f((x1, x2)), 			color=:Blues)
	scatter!([x[1]], [x[2]], label="x", color=:purple)
	scatter!([0], [0], label="x* (no constraint)", color=:green)
	show_g0_constraint && plot!([g0(t)[1] for t in 0:0.01:2π], [g0(t)[2] for t in 0:0.01:2π], lw=2, color=:red, label="g(x)=0")
	show_gradients && quiver!([x[1]], [x[2]], quiver=grads(f, x), color=:blue)
	show_gradients && quiver!([x[1]], [x[2]], quiver=grads(g, x), color=:orange)
	xlims!(x1min, x1max)
	ylims!(x2min, x2max)
end

# ╔═╡ 52bacf14-0595-11eb-33c9-dd5c40125360
∇fsoft(x) = g(x) ≤ 0.0 ? fsoft'(x) : [0.0, 0.0]

# ╔═╡ 0b93638a-0595-11eb-3096-af3cc7202aa5
begin
	contourf(x1min:0.01:x1max, x2min:0.01:x2max, (x1, x2) -> fconstr((x1, x2)), 			color=:Blues)
	
	contour!(x1min:0.1:x1max, x2min:0.1:x2max, (x1, x2) -> f((x1, x2)), 			color=:Blues)
	show_g_contour && contour!(x1min:0.1:x1max, x2min:0.1:x2max, (x1, x2) -> fsoft((x1, x2)), color=:Reds)
	scatter!([x[1]], [x[2]], label="x", color=:purple)
	scatter!([0], [0], label="x* (no constraint)", color=:green)
	show_g0_constraint && plot!([g0(t)[1] for t in 0:0.01:2π], [g0(t)[2] for t in 0:0.01:2π], lw=2, color=:red, label="g(x)=0")
	show_soft_grads && quiver!(x1min:2:x1max, (x2min:2:x2max)', quiver=(x1, x2)->-0.05.*∇fsoft((x1, x2)), color=:pink)
	xlims!(x1min, x1max)
	ylims!(x2min, x2max)
end

# ╔═╡ 0a5783e2-0596-11eb-1e38-a5da40325159


# ╔═╡ Cell order:
# ╠═49e8e6f6-0483-11eb-0a74-d94aa536c9ab
# ╟─fccca5fc-0597-11eb-1033-a5cc3b71d854
# ╠═5e4eabba-0483-11eb-09a4-31ae6c582d3e
# ╠═bd7b55ae-0599-11eb-1056-ed3682fe8160
# ╟─470048ec-0598-11eb-0d48-67357ed1f12c
# ╠═c10838a2-0599-11eb-32e3-1579a926f9b7
# ╠═41d8c8d0-0591-11eb-2cd7-3b53f91ef589
# ╠═d2c4ffe2-0485-11eb-0e20-dd52446fa3a6
# ╠═6e89822c-070b-11eb-06f3-8b8e20aaa57d
# ╠═f47ce686-0485-11eb-2a33-4bd0b150b18b
# ╟─5503c842-0591-11eb-0104-359b9ebd5e98
# ╟─47067fc8-058c-11eb-224c-6fe2e470107f
# ╟─c369c65e-0598-11eb-0a90-dd4db070c99b
# ╟─60328136-0591-11eb-1929-b7515a8c03d9
# ╟─d10b14c2-058c-11eb-3a43-7bca5d042969
# ╟─ba835a96-0598-11eb-1d54-9365e3ea337f
# ╟─681a031a-0591-11eb-2fec-8d94eca758f8
# ╟─13467b42-058d-11eb-3023-b9d3f145335b
# ╟─ca227ac2-0598-11eb-3a39-f1a90769ac52
# ╟─d8276f26-0598-11eb-1626-95f87cf6de22
# ╟─60c41848-0592-11eb-39ef-7103ae834990
# ╟─d16723ca-0598-11eb-02f4-f1c5d45095a5
# ╟─828cbe72-0483-11eb-33b2-7f4f37db889f
# ╠═95dc760a-058e-11eb-2149-898b2776079f
# ╠═aef30ee8-0597-11eb-2203-ed0f9b14f0e4
# ╟─00a009a2-0599-11eb-341f-3364198e1753
# ╟─b0cf9a40-058f-11eb-13dc-b7c35328ef2a
# ╟─13b4ad4a-0599-11eb-0043-6bfc0eb23389
# ╠═e944567a-0596-11eb-2cf1-75f196bab4d1
# ╠═02673cec-0597-11eb-2db4-e9f7d9047ac0
# ╠═7e77e628-0707-11eb-1b2f-27b5c5819260
# ╠═c04a0ffe-0707-11eb-2a74-b3b258b4aec6
# ╟─358f21ca-0599-11eb-3bfb-0bb12a1465bb
# ╟─2d7ae208-0599-11eb-1a91-277b764ad45e
# ╠═c98b4f9a-0592-11eb-2034-5f05e792cbd2
# ╠═be35a06c-0596-11eb-2988-61a79c7e6a87
# ╠═a08e7230-0591-11eb-3956-fb6c2d756494
# ╟─c2640938-0591-11eb-058c-5949d1e316d3
# ╟─81629936-0593-11eb-3fca-afa68e2ace3d
# ╠═a1402cbc-0593-11eb-33b4-09a0c654eefd
# ╠═cc4209ee-0593-11eb-3762-e76539e2a455
# ╟─a398740e-0593-11eb-2f81-c92861e080b8
# ╠═dce25666-0594-11eb-24f9-85eba4949d1e
# ╟─701efa2a-0596-11eb-2592-39eced2fb875
# ╠═7c5d381a-0596-11eb-1dfe-0b7074093cd6
# ╟─0b93638a-0595-11eb-3096-af3cc7202aa5
# ╟─7c934678-0599-11eb-1e6b-1f2c9003509a
# ╠═865bbb8e-0483-11eb-0622-b9a8a6ba5d4c
# ╠═9736f388-0483-11eb-069a-0d5ed87efb73
# ╠═f39f0106-0483-11eb-3617-5fb04388489e
# ╠═52bacf14-0595-11eb-33c9-dd5c40125360
# ╠═0a5783e2-0596-11eb-1e38-a5da40325159
