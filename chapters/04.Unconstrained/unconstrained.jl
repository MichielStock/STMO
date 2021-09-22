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

# ╔═╡ 2a6fa183-5c5a-4d50-a6d6-7a6cef635ddf
using Plots, LaTeXStrings, LinearAlgebra, PlutoUI, Zygote

# ╔═╡ 6630572c-47e6-48ac-a2b5-d21f03f317f4
module Solution
using LinearAlgebra
	export backtracking_line_search, gradient_descent, coordinate_descent, newtons_method


	"""
		backtracking_line_search(f, x, Δx, ∇f, α::Real=0.1,
							β::Real=0.7)

	Uses backtracking for finding the minimum over a line.

	Inputs:
		- f: function to be searched over a line
		- x: initial point
		- Δx: direction to search
		- ∇f: gradient of f
		- α: hyperparameter
		- β: hyperparameter

	Output:
		- t: suggested stepsize
	"""
	function backtracking_line_search(f, x, Δx, ∇f; α::Real=0.1,
							β::Real=0.7)
		@assert 0 < α < 0.5 && 0 < β < 1 "incorrect values for α and/or β"
		t = 1.0
		while f(x + t * Δx) > f(x) + α * t * ∇f(x)' * Δx
			t *= β
		end
		return t
	end


	"""
		gradient_descent(f, x₀, ∇f; α::Real=0.2, β::Real=0.7,
			ν::Real=1e-3, track=false)

	General gradient descent algorithm.

	Inputs:
		- f: function to be minimized
		- x₀: starting point
		- ∇f: gradient of the function to be minimized
		- α: parameter for btls
		- β: parameter for btls
		- ν: parameter to determine if the algorithm is converged
		- track: store the path?

	Outputs:
		- xstar: the found minimum
	"""
	function gradient_descent(f, x₀, ∇f; α::Real=0.2, β::Real=0.7,
		  ν::Real=1e-5, track=false)

		x = x₀  # initial value
		Δx = similar(x)
		track && (path=[copy(x)])
		while true
			Δx .= -∇f(x) # choose direction
			if norm(Δx) < ν
				break  # converged
			end
			t = backtracking_line_search(f, x, Δx, ∇f, α=α, β=β)
			x .+= t * Δx  # do a step
			track && push!(path, copy(x))
		end
		if track
			return x, path
		else
			return x
		end
	end

	"""
		coordinate_descent(f, x₀, ∇f; α::Real=0.2, β::Real=0.7,
			ν::Real=1e-3, track=false)

	General coordinate descent algorithm.

	Inputs:
		- f: function to be minimized
		- x₀: starting point
		- ∇f: gradient of the function to be minimized
		- α: parameter for btls
		- β: parameter for btls
		- ν: parameter to determine if the algorithm is converged
		- track: store the path?

	Outputs:
		- xstar: the found minimum
	"""
	function coordinate_descent(f, x₀::Vector, ∇f; α::Real=0.2, β::Real=0.7,
		  ν::Real=1e-5, track=false)
		x = x₀  # initial value
		Δx = zero(x)
		∇fx = similar(x)
		track && (path=[copy(x)])
		while true
			∇fx .= ∇f(x)
			i = argmax(abs.(∇fx))   # choose direction
			Δx[i] = -∇fx[i]
			if norm(∇fx) < ν
				break  # converged
			end
			t = backtracking_line_search(f, x, Δx, ∇f, α=α, β=β)  # BLS for optimal step size
			x .+= t * Δx  # do a step
			Δx[i] = 0.0  # reset
			track && push!(path, copy(x))
		end
		if track
			return x, path
		else
			return x
		end
	end

	"""
		newtons_method(f, x₀, ∇f, ∇²f; α::Real=0.2, β::Real=0.7,
			ϵ::Real=1e-7, tracker::Tracker=notrack)

	General Newton method.

	Inputs:
		- f: function to be minimized
		- x₀: starting point
		- ∇f: gradient of the function to be minimized
		- ∇²f: Hessian of the function to be minimized
		- α: parameter for btls
		- β: parameter for btls
		- ϵ: parameter to determine if the algorithm is converged
		- tracker: (bool) store the path that is followed?

	Outputs:
		- xstar: the found minimum

	"""
	function newtons_method(f, x₀, ∇f, ∇²f; α::Real=0.2, β::Real=0.7,
		  ϵ::Real=1e-5, track=false)

		x = x₀  # initial value
		Δx = similar(x)
		∇fx = similar(x)
		track && (path=[copy(x)])
		while true
			∇fx .= ∇f(x)
			Δx .= - ∇²f(x) \ ∇fx # choose direction
			λ² = - (Δx' * ∇fx)  # newton decrement
			if λ² < ϵ^2
				break  # converged
			end
			t = backtracking_line_search(f, x, Δx, ∇f, α=α, β=β)
			x .+= t * Δx  # do a step
			track && push!(path, copy(x))
		end
		if track
			return x, path
		else
			return x
		end
	end
end

# ╔═╡ 1bcccd24-0f8b-4363-949a-d498d3a41e86
using HTTP, CSV, DataFrames

# ╔═╡ 67e914ba-1b9f-11ec-3109-8b3c4fb77792
md"""
# Unconstrained convex optimization

*STMO*

**Michiel Stock**

![](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/logo.png?raw=true)
"""

# ╔═╡ 34f53c61-3315-4eb0-ba96-06fb25e507a2
md"""
## Motivation

In this chapter we will study unconstrained convex problems, i.e., problems of the form

$$\min_\mathbf{x}\, f(\mathbf{x})\,,$$

in which $f$ is *convex*. Convex optimization problems are well understood. Their most attractive property is that when a minimizer exists, the minimizer is the unique global minimizer.

Most convex optimization problems do not have a closed-form solution, with the quadratic problems of the previous chapters as a notable exception. We will hence again have to resort to descent methods to find an (arbitrary accurate) approximate solution.
"""

# ╔═╡ cbdba765-7aa5-4e08-bee1-193e6073112f
md"""
## Convex sets and functions

### Convex set

> **In words**: a set $\mathcal{C}$ is called *convex* if the line segment between any two points in $\mathcal{C}$ also lies in $\mathcal{C}$.

> **In symbols**:  a set $\mathcal{C}$ is called *convex* if, for any $\mathbf{x}, \mathbf{x}' \in \mathcal{C}$ and any $\theta \in [0, 1]$, it holds that $\theta \mathbf{x} + (1 - \theta) \mathbf{x}' \in \mathcal{C}$.

![Some convex (A & B) and non-convex sets (B & D).](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/convex_sets.png?raw=true)

Which ones do you think are convex sets?

A: $(@bind Acs CheckBox())

B: $(@bind Bcs CheckBox())

C: $(@bind Ccs CheckBox())

D: $(@bind Dcs CheckBox())
"""

# ╔═╡ d39503f3-ada3-4b38-9aec-21650a3d55b3
if Acs && !Bcs && Ccs && !Dcs
	md"Correct!"
end

# ╔═╡ 1484214a-5e5e-4a15-8e02-7739e47b18a6
md"""
### Convex functions

> **In words**:  a function $f$ is *convex* if the line segment between $(\mathbf{x}, f(\mathbf{x}))$ and $(\mathbf{x}', f (\mathbf{x}'))$ lies above the graph of $f$.

> **In symbols**: a function $f : \mathbb{R}^n\rightarrow \mathbb{R}$ is *convex* if
> - dom($f$) is convex
> - for any $\mathbf{x}, \mathbf{x}' \in \text{dom}(f)$ and any $\theta \in [0, 1]$, it holds that $f(\theta \mathbf{x} + (1-\theta)\mathbf{x}') \leq\theta f(\mathbf{x}) +(1-\theta)f(\mathbf{x}')$.

Below is an example of a convex function.
"""

# ╔═╡ 7c11ebc0-f2f4-4d50-b691-533e5b996c88
md"
![Some convex (A & B) and non-convex sets (B & D).](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/convex_functions.png?raw=true)

Which ones do you think are convex functions?

A: $(@bind Acf CheckBox())

B: $(@bind Bcf CheckBox())

C: $(@bind Ccf CheckBox())
"

# ╔═╡ c7e1ee9e-6a00-47ca-a540-84029bba301a
if Acf && !Bcf && Ccf
	md"Correct!"
end

# ╔═╡ 421c5ff4-d7df-4340-92ab-a38f25a67090
md"""

From the definition, it follows that:

- If the function is differentiable, then $f(\mathbf{x})\geq f(\mathbf{x}')+\nabla f(\mathbf{x}')^\top(\mathbf{x}-\mathbf{x}')$ for all $\mathbf{x}$ and $\mathbf{x}' \in \text{dom}(f)$. **The first-order Taylor approximation is a global underestimator of $f$.**
- If the function is twice differentiable, then $\nabla^2 f(\mathbf{x})\succeq 0$ for any $\mathbf{x}\in\text{dom}(f)$.

Convex functions frequently arise:

- If $f$ and $g$ are both convex, then $m(x)=\max(f(x), g(x))$ and $h(x)=f(x)+g(x)$ are also convex.
- If $f$ and $g$ are convex functions and $g$ is non-decreasing over a univariate domain, then $h(x)=g(f(x))$ is convex. Example: $e^{f(x)}$ is convex if $f({x})$ is convex.
- If $f$ is concave and $g$ is convex and non-increasing over a univariate domain, then $h(x)=g(f(x))$ is convex.

Note, the convexity of expected value in probability theory gives rise to *Jensen's inequality*. For any convex function $\varphi$, if holds that

$$\varphi(\mathbb{E}[X]) \leq\mathbb{E}[\varphi(X)]\,.$$

This implies for example that the square of an expected value of quantity is never greater than the expected square of that quantity.
"""

# ╔═╡ 3c5b595f-4b63-4036-9d01-7fb1daa60029
md"""
### Strongly convex functions

> **In words**: a function $f$ is called *strongly convex* if it is at least as convex as a quadratic function.

> **In symbols**: a $f$ is called *strongly $m$-convex* (with $m>0$) if the function $f_m(\mathbf{x}) = f(\mathbf{x}) - \frac{m}{2}||\mathbf{x}||_2$ is convex.

If the first- and second order derivatives exists, a strongly $m$-convex function satisfies:

-  $f(\mathbf{x}') \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{x}'-\mathbf{x}) + \frac{m}{2}||\mathbf{x}'-\mathbf{x}||_2$
-  $\nabla^2 f(\mathbf{x})-mI\succeq 0$ (all eigenvalues of the Hessian are greater than $m$)

If a function is $m$-strongly convex, this also implies that there exists an $M>m$ such that

$$\nabla^2 f(\mathbf{x}) \preceq MI\,.$$

Stated differently, for strongly convex functions the exist both a quadratic function with a smaller as well as a lower local curvature.

![For strongly convex functions, it holds that there are two constants $m$ and $M$ such that $mI\preceq\nabla^2 f(\mathbf{x}) \preceq MI$. ](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/strong_convexity.png?raw=true)
"""

# ╔═╡ a48204c9-9bbb-48bc-b1c1-825431b0d0cc
md"""
## Toy examples

To illustrate the algorithms, we introduce two toy functions to minimize:

- simple quadratic problem:
$$f(x_1, x_2) = \frac{1}{2} (x_1^2 +\gamma x_2^2)\,,$$
where $\gamma$ determines the condition number.
- a non-quadratic function:
$$f(x_1, x_2) = \log(e^{x_1 +3x_2-0.1}+e^{x_1 -3x_2-0.1}+e^{-x_1 -0.1})\,.$$
"""

# ╔═╡ 0b86ef87-db56-48ce-878f-81d8c65294b5
md"""
## General descent methods (recap)

Convex functions are usually minimized using descent methods. Again, line search is often used as a subroutine.

The outline of a general descent algorithm is given in the following pseudocode.

> **input** starting point $\mathbf{x}\in$ **dom** $f$.
>
> **repeat**
>
>>    1. Determine a descent direction $\Delta \mathbf{x}$.
>>    2. *Line seach*. Choose a step size $t>0$.
>>    3. *Update*. $\mathbf{x}:=\mathbf{x}+t\Delta \mathbf{x}$.
>
> **until** stopping criterion is satisfied.
>
> **output** $\mathbf{x}$

The specific optimization algorithms are hence determined by:

- method for determining the search direction $\Delta \mathbf{x}$, this is almost always based on the gradient of $f$
- method for choosing the step size $t$, may be fixed or adaptive
- the criterion used for terminating the descent, usually the algorithm stops when the improvement is smaller than a predefined value

"""

# ╔═╡ de63922b-584a-4bb9-9d9b-79e75b60cb1d
md"""
## Backtracking line search

For quadratic optimization, as covered in Chapter 1, the optimal step size could be computed in closed form. In the general case, only an approximately optimal step size is used.
"""

# ╔═╡ 26063cc6-1ac3-4149-9e96-7ffcae53d57d
md"""
### Exact line search

As a subroutine of the general descent algorithm a line search has to be performed. A value for $t$ is chosen to minimize $f$ along the ray $\{\mathbf{x}+t\Delta \mathbf{x} \mid t\geq0\}$:

$$t = \text{arg min}_{s\geq0}\ f(\mathbf{x}+s\Delta \mathbf{x})\,.$$

Exact line search is used when the cost of solving the above minimization problem is small compared to the cost of calculating the search direction itself. This is sometimes the case when an analytical solution is available.
"""

# ╔═╡ 9d4f5f9c-0cbb-4f48-be7f-0365fa8c0cbb
md"""
### Inexact line search

Often, the descent methods work well when the line search is done only approximately. This is because the computational resources are better spent to performing more *approximate* steps in the different directions because the direction of descent will change anyway.

Many methods exist for this, we will consider the *backtracking line search* (BTLS), described by the following pseudocode.

> **input** starting point $\mathbf{x}\in$ **dom** $f$, descent direction $\Delta \mathbf{x}$, gradient $\nabla f(\mathbf{x})$,  $\alpha\in(0,0.5)$ and $\beta\in(0,1)$.
>
>  $t:=1$
>
>**while** $f(\mathbf{x}+t\Delta \mathbf{x}) > f(x) +\alpha t \nabla f(\mathbf{x})^\top\Delta \mathbf{x}$
>
>>    $t:=\beta t$
>
>
>**output** $t$
"""

# ╔═╡ 64d464af-3403-4ea9-8fd5-983792d7b125
md"""
The backtracking line search has two parameters:

-  $\alpha$: fraction of decrease in $f$ predicted by linear interpolation we accept
-  $\beta$: reduction of the step size in each iteration of the BTLS
-  typically, $0.01 \leq \alpha \leq 0.3$ and $0.1 \leq \beta < 1$

![Illustration of the backtracking line search.](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/btls.png?raw=true)
"""

# ╔═╡ fc9bddb6-69e7-44ca-820e-b2b67770067b
md"**Assignment 1**

1. Complete the code for the backtracking line search
2. Use this function find the step size $t$ to (approximately) minimize $f(x) = x^2 - 2x - 5$ starting from the point $0$. Choose a $\Delta x=10$.
"

# ╔═╡ af3f1a9c-32c6-4e4b-8aeb-feb2cc99160d
"""
    backtracking_line_search(f, x, Δx, grad_f, α::Real=0.1,
                        β::Real=0.7)

Uses backtracking for finding the minimum over a line.

Inputs:
    - f: function to be searched over a line
    - x: initial point
    - Δx: direction to search
    - grad_f: gradient of f
    - α
    - β

Output:
    - t: suggested stepsize
"""
function backtracking_line_search(f, x, Δx, grad_f; α::Real=0.1,
                        β::Real=0.7)
    @assert 0 < α < 0.5 && 0 < β < 1 "incorrect values for α and/or β"
    t = 1.0
    while missing  # complete
        missing  # complete
    end
    return t
end

# ╔═╡ 498c9140-1bd0-4b73-a6f0-57898a80164d
fun(x) = missing

# ╔═╡ 286c3b12-cb4d-489a-b4f6-b5a3aafedf9a
grad_fun(x) = missing

# ╔═╡ 6fc90c99-4782-4881-bb87-18b0723ff862
#backtracking_line_search(...)

# ╔═╡ 64a52a2f-65ea-4efe-8f63-826b5cb9745b
md"**Question 1**

Describe the effect of $\alpha$, $\beta$ and $\Delta \mathbf{x}$. How can you perform a more precise search?
"

# ╔═╡ 045e627e-09bf-4589-bada-2d184d68977a
md"""
## Gradient descent

A natural choice for the search direction is the negative gradient: $\Delta \mathbf{x} = -\nabla f(\mathbf{x})$. This algorithm is called the *gradient descent algorithm*."""

# ╔═╡ 723d9b31-eb75-49c1-a33a-9e105d3ccdfb
md"""
### General gradient descent algorithm

>**input** starting point $\mathbf{x}\in$ **dom** $f$.
>
>**repeat**
>
>>    1. *Choose direction*. $\Delta \mathbf{x} := -\nabla f(\mathbf{x})$.
>>    2. *Line seach*. Choose a step size $t$ via exact or backtracking line search.
>>    3. *Update*. $\mathbf{x}:=\mathbf{x}+t\Delta \mathbf{x}$.
>
>**until** stopping criterion is satisfied.
>
>**output** $\mathbf{x}$

The stopping criterion is usually of the form $||\nabla f(\mathbf{x})||_2 \leq \nu$.
"""

# ╔═╡ 533893ac-b89b-48c0-afa1-2b1284fda494
md"""
### Convergence analysis

The notion of strongly convexity allows us to bound the function $f$ by two quadratic functions. As such we can reuse the convergence analysis of the previous chapter.

If $f$ is strongly convex (constants $m$ and $M$ exist such that $mI\prec \nabla^2 f(\mathbf{x})\prec MI$), it holds that $f(\mathbf{x}^{(k)}) - p^*\leq \varepsilon$ after at most

$$\frac{\log((f(\mathbf{x}^{(0)}) - p^*)/\varepsilon)}{\log(1/c)}$$
iterations, where $c =1-\frac{m}{M}<1$.

We conclude:

- The number of steps needed for a given quality is proportional to the logarithm of the initial error.
- To increase the accuracy with an order of magnitude, only a few more steps are needed.
- Convergence is again determined by the *condition number* $M/m$. Note that for large condition numbers: $\log(1/c)=-\log(1-\frac{m}{M})\approx m/M$, so the number of required iterations increases linearly with increasing $M/m$.
"""

# ╔═╡ 0bca2788-018b-4b54-b910-3d19fed02788
md"**Assignment 2**

1. Complete the implementation of the gradient descent method.
2. Plot the paths for the two toy problems. Use $\mathbf{x}^{(0)}=[10,1]^\top$ for the quadratic function and $\mathbf{x}^{(0)}=[-0.5,0.9]^\top$ for the non-quadratic function as starting points.
3. Analyze the convergence.
"

# ╔═╡ 33a0c8ac-ba23-4ac2-aafe-c5622b7e2255
"""
    gradient_descent(f, x₀, grad_f; α::Real=0.2, β::Real=0.7,
        ν::Real=1e-3)

General gradient descent algorithm.

Inputs:
    - f: function to be minimized
    - x₀: starting point
    - grad_f: gradient of the function to be minimized
    - α: parameter for btls
    - β: parameter for btls
    - ν: parameter to determine if the algorithm is converged

Outputs:
    - xstar: the found minimum
"""
function gradient_descent(f, x₀, grad_f; α::Real=0.2, β::Real=0.7,
      ν::Real=1e-7)

    x = x₀  # initial value
    Δx = similar(x)
    nsteps = 0
    while true
        missing # choose direction
        if missing
            break  # converged
        end
        t = missing
        missing # do a step
        nsteps += 1
    end
    println("converged after $nsteps steps")
    return x
end

# ╔═╡ 3f2d01cc-b0e2-4ee1-9e90-eec8a179c447


# ╔═╡ 186ee3e3-a36d-440f-bfa1-27efdb80ce1f


# ╔═╡ d3c33ef9-29dc-45e9-aae6-e7bc062971fc
md"""
## Steepest descent

Optimize the first-order Taylor approximation of a function:

$$f(\mathbf{x}+\mathbf{v}) \approx \hat{f}(\mathbf{x}+\mathbf{v}) =f(\mathbf{x}) +\nabla f(\mathbf{x})^\top \mathbf{v}\,.$$

The linear approximation $\hat{f}$ can be made arbitrary negative if we can freely choose $\mathbf{v}$! We have to constrain the *norm* of $\mathbf{v}$.

### Vector norms

A norm on $\mathbb{R}^n$ is a function $||\cdot||:\mathbb{R}^n\rightarrow \mathbb{R}$ with the following properties:

-  $||\mathbf{x}||>0$, for any $\mathbf{x}\in\mathbb{R}^n$
-  $||\mathbf{x}+\mathbf{y}|| \leq ||\mathbf{x}||+||\mathbf{y}||$, for any $\mathbf{x}, \mathbf{y}\in\mathbb{R}^n$
-  $||\lambda \mathbf{x}|| = |\lambda|\, ||\mathbf{x}||$ for any $\lambda \in\mathbb{R}$ and any $\mathbf{x}\in\mathbb{R}^n$
-  $||\mathbf{x}||=0$ if and only if $\mathbf{x}=0$

For example, for any $\mathbf{x}\in\mathbb{R}^n$ and $p\leq 1$:

$$||\mathbf{x}||_p = \left(\sum_{i=1}^n |x_i|^p\right)^\frac{1}{p}\,.$$

$||\cdot||_1$ is often called the $L_1$ norm and $||\cdot||_2$ the $L_2$ norm.

Consider $P\in \mathbb{R}^{n\times n}$ such that $P\succ 0$. The  corresponding *quadratic norm*:

$$||\mathbf{z}||_P = (\mathbf{z}^\top P\mathbf{z})^\frac{1}{2}=||P^\frac{1}{2}\mathbf{z}||_2\,.$$

The matrix $P$ can be used to encode prior knowledge about the scales and dependencies in the space that we want to search.

### Dual norm

Let $|| \cdot ||$ be a norm on $\mathbb{R}^n$. The associated dual norm:
$$||\mathbf{z}||_*=\sup_{\mathbf{x}} \{\mathbf{z}^\top\mathbf{x}\mid ||\mathbf{x}||\leq 1\}\,.$$

Examples:

- the dual norm of $||\cdot||_1$ is $||\cdot||_\infty$;
- the dual norm of $||\cdot||_2$ is $||\cdot||_2$;
- the dual norm of $||\cdot||_P$ is defined by $||\mathbf{z}||_*=||P^{-\frac{1}{2}}\mathbf{z}||$.
"""

# ╔═╡ d6fcda50-4bd3-49b5-bd2c-11c278e3bb31
md"""
### Steepest descent directions

**Normalized steepest descent direction**:

$$\Delta x_\text{nsd} = \text{arg min}_\mathbf{v}\, \{\nabla f(\mathbf{x})^T \mathbf{v} \mid ||\mathbf{v}||_1\leq 1 \}\,.$$

**Unnormalized steepest descent direction**:

$$\Delta x_\text{sd} = ||\nabla f(\mathbf{x})||_\star \Delta x_\text{nsd} \,.$$

Note that we have
$$\nabla f(\mathbf{x})^\top \Delta x_\text{sd} = ||\nabla f(\mathbf{x})||_\star \nabla f(\mathbf{x})^\top\Delta x_\text{nsd} = -||\nabla f(\mathbf{x})||^2_\star\,,$$

so this is a valid descent method.

![Illustration of some descent directions based on different norms.](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/sd_gradients.png?raw=true)

"""

# ╔═╡ d6c28b2e-24d0-41dd-931e-586f4a906a90
md"""
### Coordinate descent algorithm

Using the $L_1$ norm results in coordinate descent. For every iteration in this algorithm, we descent in the direction of the dimension where the absolute value of the gradient is largest.

>**input** starting point $\mathbf{x}\in$ **dom** $f$.
>
>**repeat**
>
>>    1. *Direction*. Choose $i$ such that $|\nabla f(\mathbf{x})_i|$ is maximal.
>>    2. *Choose direction*. $\Delta \mathbf{x} := -\nabla f(\mathbf{x})_i \mathbf{e}_i$
>>    3. *Line seach*. Choose a step size $t$ via exact or backtracking line search.
>>    4. *Update*. $\mathbf{x}:=\mathbf{x}+t\Delta \mathbf{x}$.
>
>**until** stopping criterion is satisfied.
>
>**output** $\mathbf{x}$

Here, $\mathbf{e}_i$ is the $i$-th basic vector.

The stopping criterion is usually of the form $||\nabla f(\mathbf{x})||_2 \leq \nu$.

Coordinate descent optimizes every dimension in turn, for this reason it is sometimes used in minimization problems which enforce sparseness (e.g. LASSO regression).

> *Optimizing one dimension at a time is usually a poor strategy. This is because different dimensions are often related.*
"""

# ╔═╡ 0db2b747-5d0c-4eac-a722-91b8cecf2bd8
md"**Assignment 3**

1. Complete the implementation of the coordinate descent method.
2. Plot the paths for the two toy problems. Use the same stating points as before.
3. Analyze the convergence."

# ╔═╡ f38f6ab4-6718-4fe0-bba1-a2fe04592bdd
"""
    coordinate_descent(f, x₀, grad_f; α::Real=0.2, β::Real=0.7,
        ν::Real=1e-3)

General coordinate descent algorithm.

Inputs:
    - f: function to be minimized
    - x₀: starting point
    - grad_f: gradient of the function to be minimized
    - α: parameter for btls
    - β: parameter for btls
    - ν: parameter to determine if the algorithm is converged

Outputs:
    - xstar: the found minimum
"""
function coordinate_descent(f, x₀::Vector, grad_f; α::Real=0.2, β::Real=0.7,
      ν::Real=1e-7)
    x = x₀  # initial value
    Δx = zero(x)
    nsteps = 0
    while true
        missing   # choose direction

        if missing
            break  # converged
        end
        missing  # BLS for optimal step size
        missing  # do a step
        nsteps += 1
    end
    println("converged after $nsteps steps")
    return x
end

# ╔═╡ 599b2161-f45f-49df-85ca-9a9649ed7f01


# ╔═╡ e95d1fd3-83f9-4001-8c23-01cb45630daa


# ╔═╡ 3c3561b2-03d8-46a1-910a-ba5b2b43f082
md"""
## Newton's method

### The Newton step

In Newton's method the descent direction is chosen as

$$\Delta \mathbf{x}_\text{nt} = -(\nabla^2f(\mathbf{x}))^{-1} \nabla f(\mathbf{x})\,,$$

which is called the *Newton step*.

If $f$ is convex, then $\nabla^2f(\mathbf{x})$ is positive definite and

$$\nabla f(\mathbf{x})^\top \Delta \mathbf{\mathbf{x}}_\text{nt} \leq 0\,,$$

hence the Newton step is a descent direction unless $\mathbf{x}$ is optimal.

This Newton step can be motivated in several ways.

**Minimizer of a second order approximation**

The second order Taylor approximation $\hat{f}$ of $f$ at $\mathbf{x}$ is

$$f(\mathbf{x}+\mathbf{v})\approx\hat{f}(\mathbf{x}+\mathbf{v}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{v} + \frac{1}{2} \mathbf{v}^\top \nabla^2 f(\mathbf{x}) \mathbf{v}\,$$

which is a convex quadratic function of $\mathbf{v}$, and is minimized when $\mathbf{v}=\Delta \mathbf{x}_\text{nt}$.

This quadratic model will be particularly accurate when $\mathbf{x}$ is close to $\mathbf{x}^\star$.

**Steepest descent direction in Hessian norm**

The Newton step is the steepest descent step if a quadratic norm using the Hessian is used, i.e.

$$||\mathbf{u}||_{\nabla^2f(\mathbf{x})}=(\mathbf{u}^\top(\nabla^2f(\mathbf{x}))^{-1}\mathbf{u})^\frac{1}{2}\,.$$

**Affine invariance of the Newton step**

> *A consistent algorithm should give the same results independent of the units in which quantities are measured.*  ~ Donald Knuth

The Newton step is independent of linear or affine changes of coordinates. Consider a non-singular $n\times n$ transformation matrix $T$. If we apply a coordinate transformation $\mathbf{x}=T\mathbf{y}$ and define $\bar{f}(\mathbf{y}) = f(\mathbf{x})$, then

$$\nabla \bar{f}(\mathbf{y}) = T^\top\nabla f(\mathbf{x})\,,\quad \nabla^2 \bar{f}(\mathbf{y}) = T^\top\nabla^2f(\mathbf{x})T\,.$$

As such it follows that

$$\mathbf{x} + \Delta \mathbf{x}_\text{nt} = T (\mathbf{y} + \Delta \mathbf{y}_\text{nt})\,.$$
"""

# ╔═╡ 46e6c9a2-6fae-4abc-87f8-d423a63df6ad
md"""
**Questions 2**

Does scaling and rotation affect the working of gradient descent and coordinate descent?

### Newton decrement

The Newton decrement is defined as

$$\lambda(\mathbf{x})  = (\nabla f(\mathbf{x})^\top\nabla^2 f(x)^{-1}\nabla f(\mathbf{x}))^{1/2}\,.$$

This can be related to the quantity $f(\mathbf{x})-\inf_\mathbf{y}\ \hat{f}(\mathbf{y})$:

$$f(\mathbf{x})-\inf_\mathbf{y}\ \hat{f}(\mathbf{y}) = f(\mathbf{x}) - \hat{f}(\mathbf{x} +\Delta \mathbf{x}_\text{nt}) = \frac{1}{2} \lambda(\mathbf{x})^2\,.$$

Thus $\frac{1}{2} \lambda(\mathbf{x})^2$ is an estimate of $f(\mathbf{x}) - p^*$, based on the quadratic approximation of $f$ at $\mathbf{x}$.

### Pseudocode of Newton's algortihm

>**input** starting point $\mathbf{x}\in$ **dom** $f$.
>
>**repeat**
>
>>    1. Compute the Newton step and decrement $\Delta \mathbf{x}_\text{nt} := -\nabla^2f(\mathbf{x})^{-1} \nabla f(\mathbf{x})$; $\lambda^2:=\nabla f(\mathbf{x})^\top\nabla^2 f(\mathbf{x})^{-1}\nabla f(\mathbf{x})$.
>>    2. *Stopping criterion* **break** if $\lambda^2/2 \leq \epsilon$.
>>    3. *Line seach*. Choose a step size $t$ via exact or backtracking line search.
>>    4. *Update*. $\mathbf{x}:=\mathbf{x}+t\Delta \mathbf{x}_\text{nt}$.
>
>**output** $\mathbf{x}$

The above algorithm is sometimes called the *damped* Newton method, as it uses a variable step size $t$. In practice, using a fixed step also works well. Here, one has to consider the computational cost of using BTLS versus performing a few extra Newton steps to attain the same accuracy.

See below for the paths of Newton's algorithm on the quadratic and non-quadratic functions. Note that the quadratic problem is solved exactly in one step.

![](https://github.com/MichielStock/STMO/blob/master/chapters/04.Unconstrained/Figures/newtons_method.png?raw=true)

The following figure shows the convergence of Newton's algorithm on the quadratic and non-quadratic functions. Note that the quadratic problem is solved exactly in one step.
"""

# ╔═╡ 5015de28-d9f6-4a29-a380-4f059d8b9c66
md"### Convergence analysis

Iterations in Newton's method fall into two stages:

- *damped Newton phase* $(t < 1)$ until $||\nabla f(\mathbf{x})||_2 \leq \eta$
- *pure Newton phase* $(t = 1)$: quadratic convergence

After a sufficiently large number of iterations, the number of correct digits doubles at each iteration.
"

# ╔═╡ 2185cb3e-2a81-43fb-845c-e786ec23d16b
md"**Assignment 4**

1. Complete the code for Newton's method.
2. Find the minima of the two toy problems. Use the same starting points as for gradient descent.
"

# ╔═╡ a6174578-3c81-4aec-a47d-b48d58e7a120
"""
    newtons_method(f, x₀, Df, DDf; α::Real=0.2, β::Real=0.7,
        ϵ::Real=1e-7)

General Newton method.

Inputs:
    - f: function to be minimized
    - x₀: starting point
    - Df: gradient of the function to be minimized
    - DDf: Hessian of the function to be minimized
    - α: parameter for btls
    - β: parameter for btls
    - ϵ: parameter to determine if the algorithm is converged

Outputs:
    - xstar: the found minimum
"""
function newtons_method(f, x₀, Df, DDf; α::Real=0.2, β::Real=0.7,
      ϵ::Real=1e-5)

    x = x₀  # initial values
    # preallocation
    Δx = similar(x)
    Dfx = similar(x)
    nsteps = 0
    while true
        missing # choose direction
        λ² = missing  # newton decrement
        if λ² < ϵ
            break  # converged
        end
        missing
        missing  # do a step
        nsteps += 1
    end
    println("converged after $nsteps steps")
    return x
end

# ╔═╡ ed7823a9-17c9-4402-a870-528648053bb4


# ╔═╡ c635dade-3c56-4edb-8446-a52e260a66a7


# ╔═╡ 3a367342-3742-4c7d-ba2a-6345414bf283
md"### Summary Newton's method

- Convergence of Newton's algorithm is rapid and quadratic near $\mathbf{x}^\star$.
- Newton's algorithm is affine invariant, e.g. invariant to choice of coordinates or condition number.
- Newton's algorithm scales well with problem size. Computationally, computing and storing the Hessian might be prohibitive.
- The hyperparameters $\alpha$ and $\beta$  of BTLS do not influence the performance much.
"

# ╔═╡ c69f3e77-eaa9-44fa-9274-5d78e559fa1b
md"## Quasi-Newton methods

Quasi-Newton methods try to emulate the success of the Newton method, but without the high computational burden of constructing the Hessian matrix every step. One of the most popular quasi-Newton algorithms is the *Broyden-Fletcher-Goldfarb-Shanno* (BFGS) algorithm. Here, the Hessian is approximated by a symmetric rank-one matrix."

# ╔═╡ 8467ce85-2d9b-417d-86e4-6f5f6c0c67c2
md"""## Exercise: logistic regression

Consider the following problem: we have a dataset of $n$ instances: $T=\{(\mathbf{x}_i, y_i)\mid i=1\ldots n\}$. Here $\mathbf{x}_i\in \mathbb{R}^p$ is a $p$-dimensional feature vector and $y_i\in\{0,1\}$ is a binary label. This is a binary classification problem, we are interested in predicting the label of an instance based on its feature description. The goal of logistic regression is to find a function $f(\mathbf{x})$ that estimates the conditional probability of $Y$:

$$\mathcal{P}(Y=1 \mid \mathbf{X} = \mathbf{x})\,.$$

We will assume that this function $f(\mathbf{x})$ is of the form

$$f(\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x})\,,$$

with $\mathbf{w}$ a vector of parameters to be learned and $\sigma(.)$ the logistic map:

$$\sigma(t) = \frac{e^{t}}{1+e^{t}}=\frac{1}{1+e^{-t}}\,.$$

It is easy to see that the logistic mapping will ensure that $f(\mathbf{x})\in[0, 1]$, hence $f(\mathbf{x})$ can be interpreted as a probability.

Note that

$$\frac{\text{d}\sigma(t)}{\text{d}t} = (1-\sigma(t))\sigma(t)\,.$$

To find the best weights that separate the two classes, we can use the following structured loss function:

$$\mathcal{L}(\mathbf{w})=-\sum_{i=1}^n[y_i\log(\sigma(\mathbf{w}^\top\mathbf{x}_i))+(1-y_i)\log(1-\sigma(\mathbf{w}^\top\mathbf{x}_i))] +\lambda \mathbf{w}^\top\mathbf{w}\,.$$

Here, the first part is the cross entropy, which penalizes disagreement between the prediction $f(\mathbf{x}_i)$ and the true label $y_i$, while the second term penalizes complex models in which $\mathbf{w}$ has a large norm. The trade-off between these two components is controlled by $\lambda$, a hyperparameter. In the course *Predictive modelling* of Willem Waegeman it is explained that by carefully tuning this parameter one can obtain an improved performance. **In this project we will study the influence $\lambda$ on the convergence of the optimization algorithms.**

> **Warning**: for this project there is a large risk of numerical problems when computing the loss function. This is because in the cross entropy $0\log(0)$ should by definition evaluate to its limit value of $0$. Numpy will evaluate this as `nan`. Use the provided function `cross_entropy` which safely computes $-\sum_{i=1}^n[y_i\log(\sigma_i)+(1-y_i)\log(1-\sigma_i)]$.


**Data overview**

Consider the data set in the file `BreastCancer.csv`. This dataset contains information about 569 female patients diagnosed with breast cancer. For each patient it was recorded wether the tumor was benign (B) or malignant (M), this is the response variable. Each tumor is described by 30 features, which encode some information about the tumor. We want to use logistic regression with regularization to predict wether a tumor is benign or malignant based on these features.
"""

# ╔═╡ 62e9e6ba-b236-48c6-ae17-955444547693
md"**Assignments**

1. Derive and implement the loss function for logistic loss, the gradient and the Hessian of this loss function. These functions have as input the parameter vector $\mathbf{w}$, label vector $\mathbf{y}$, feature matrix $\mathbf{X}$ and $\lambda$. The logistic map and cross-entropy is already provided for you.
2. Consider $\lambda=0.1$, find the optimal parameter vector for this data using gradient descent, coordinate descent and Newton's method. Use standardized features. For each algorithm, give the number of steps the algorithm performed and the running time (use the `@elapsed`). Compare the loss for each of parameters obtained by the different algorithms.
3. How does regularization influence the optimization? Make a separate plot for gradient descent, coordinate descent and Newton's method with the the value of the loss as a function of the iteration of the given algorithm. Make separate the different methods and plot the convergence for $\lambda = [10^{-3}, 10^{-1}, 1, 10, 100]$. Does increased regularization make the optimization go faster or slower? Why does this make sense?
"

# ╔═╡ 851bfa30-1a9b-414c-8584-4e544b84a3ab
logistic_map(x) = 1 / (1.0 + exp(-x));

# ╔═╡ 9abb8ced-16e8-413d-91cd-7f5b108ad044
σ(x) = logistic_map(x)

# ╔═╡ 4c93c629-fbbf-4981-ab21-74eab8c29265
cross_entropy(l::Bool, p) = l ? -log(p) : -log(1.0 - p)

# ╔═╡ c994bc37-7e7f-4ca3-ad2a-502df6ff48c5
"""
    logistic_loss(w, y, X; λ)

Implement the logistic loss
returns a scalar
"""
function logistic_loss(w, y, X; λ)
    return missing
end

# ╔═╡ e9135d62-870b-4c31-b772-10f50c5e8059
"""
    grad_logistic_loss(w, y, X; λ)

Implement the gradient of the logistic loss
returns a column vector
"""
function grad_logistic_loss(w, y, X; λ)
    return missing
end

# ╔═╡ ada77919-3960-4d76-9d72-0a8e343b41dd
"""
    hess_logistic_loss(w, y, X; λ)
Implement the Hessian of the logistic loss
returns a matrix
"""
function hess_logistic_loss(w, y, X; λ)
    return missing
end

# ╔═╡ be59e014-f7ce-4cb3-8cce-bdaa2f1bb2e1
md"**Assignment 2**

Use gradient descent, coordinate descent and Newton's method to find the parameters of the logistic model ($\lambda=0.1$).
"

# ╔═╡ 5a891bd7-1a9d-40fb-a8fc-9d1fa036e81c


# ╔═╡ f78663c8-1581-48e8-9475-a9a15220e15f
md"**Assignment 3**

Make a plot for each of the four optimization method in which you show the convergence for $\lambda = [10^{-3}, 10^{-1}, 1, 10, 100]$.
"

# ╔═╡ 75deabf3-3a4b-424d-9231-2ddec5aaf5eb


# ╔═╡ 82838573-5cb4-4a4a-83be-af7ec0564b7a


# ╔═╡ f16d253f-aa84-4de9-81ac-f7a8daca6f35


# ╔═╡ 1cabff7d-2fd2-4548-94ed-2d3f49496f4b
md"
## References

- Boyd, S. and Vandenberghe, L., '*[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)*'. Cambridge University Press (2004)
"

# ╔═╡ 86140ad2-6204-4ef4-8644-3c72ba77fbe3
begin
	myblue = "#304da5"
	mygreen = "#2a9d8f"
	myyellow = "#e9c46a"
	myorange = "#f4a261"
	myred = "#e76f51"
	myblack = "#50514F"

	mycolors = [myblue, myred, mygreen, myorange, myyellow]
end;

# ╔═╡ 73b10099-a769-48af-bb0b-6d3aa785e612
let
	f(x) = 0.1x^4 - 2x + x^2

plot(f, -4:0.1:4, xlabel="\$f(x)\$", color=myblue, lw=2)

x, x′ = -3.5, 2.75
scatter!([x, x′], f.([x, x′]), label="\$x, x'\$", color=mygreen)
plot!([x, x′], f.([x, x′]), label="\$(1-\\theta)f(x)+\\theta f(x')\$", color=myred, lw=2)
end

# ╔═╡ 17c12a43-b9f9-42b0-886d-92dddf32e767
begin
	
	fquadr((x1, x2); γ=10.0) = 0.5(x1^2 + γ * x2^2);
	grad_fquadr((x1, x2); γ=10.0) = [x1, γ * x2];
	hess_fquadr((x1, x2); γ=10.0) = [1 0; 0 γ];
	
	fnonquadr((x1, x2)) = log(exp(x1+3x2-0.1) + exp(x1-3x2-0.1)+exp(-x1-0.1));
	grad_fnonquadr(x) = Zygote.gradient(fnonquadr, x)[1];
	hess_fnonquadr(x) = Zygote.hessian(fnonquadr, x);

end

# ╔═╡ 34786d52-19e2-4c4b-ac7c-04316a400216
pq = contour(-10:0.1:10, -5:0.1:5, (x1, x2) -> fquadr((x1, x2)), xlabel="\$ x_1 \$",
                ylabel="\$ x_2 \$", title="quadratic")

# ╔═╡ 06eab55d-a514-424b-9270-a7a0e82f0847
pnq = contour(-2:0.1:2, -1:0.1:1, (x1, x2) -> fnonquadr((x1, x2)), xlabel="\$ x_1 \$",
                ylabel="\$ x_2 \$", title="non-quadratic")

# ╔═╡ 5d1634c3-deff-44ae-884d-0d250ce8879c
begin
	
	# starting point
	x0q = [9.0, 2.0];
	x0nq = [-1.0, 0.75];

	# GRADIENT DESCENT
	xstarq, path_qgd = Solution.gradient_descent(fquadr, copy(x0q), grad_fquadr, track=true);
	xstarnq, path_nqgd = Solution.gradient_descent(fnonquadr, copy(x0nq), grad_fnonquadr, track=true);
	
	# COORDINATE DESCENT

	xstarq, path_qcd = Solution.coordinate_descent(fquadr, copy(x0q), grad_fquadr, track=true);

	xstarq, path_nqcd = Solution.coordinate_descent(fnonquadr, copy(x0nq), grad_fnonquadr, track=true);

	# NEWTON

	xstarq, path_qnew = Solution.newtons_method(fquadr, copy(x0q), grad_fquadr, hess_fquadr, track=true);
	xstarq, path_nqnew = Solution.newtons_method(fnonquadr, copy(x0nq), grad_fnonquadr, hess_fnonquadr, track=true);

end;

# ╔═╡ 91242142-a86f-4357-b6bd-3496d94ba7a4
begin
	plot!(pq, first.(path_qgd), last.(path_qgd), label="GD", lw=2, color=myorange)
	plot!(pq, first.(path_qcd), last.(path_qcd), label="CD", lw=2, color=myred)
	plot!(pq, first.(path_qnew), last.(path_qnew), label="Newton", lw=2, color=myblue)
end

# ╔═╡ 16aa0de1-1dfc-4168-839f-35b86eca4269
begin
	plot!(pnq, first.(path_nqgd), last.(path_nqgd), label="GD", lw=2, color=myorange)
	plot!(pnq, first.(path_nqcd), last.(path_nqcd), label="CD", lw=2, color=myred)
	plot!(pnq, first.(path_nqnew), last.(path_nqnew), label="Newton", lw=2, color=myblue)
end

# ╔═╡ 07d59dd2-1d0b-4795-a2c4-72d2b522c81b
let
	
	fnqexact = fnonquadr(xstarq)
	nonquadrerr = x -> abs(fnonquadr(x) .- fnqexact);
	
	p1 = plot(fquadr.(path_qgd), label="GD", yscale=:log10, ylabel="\$ f(x) - f(x^\\star)\$", title="quadratic", ylim=(1e-15, 10), color=myorange);
	plot!(p1, fquadr.(path_qcd), color=myred, label="CD");
	plot!(p1, fquadr.(path_qnew), color=myblue, label="Newton");
	xlabel!("iteration + 1")
	
	p2 = plot(nonquadrerr.(path_nqgd), label="GD", yscale=:log10, ylabel="\$ f(x) - f(x^\\star)\$", title="non-quadratic", ylim=(1e-15, 10), color=myorange);
	plot!(p2, nonquadrerr.(path_nqcd), color=myred, label="CD");
	plot!(p2, nonquadrerr.(path_nqnew), color=myblue, label="Newton");
	xlabel!("iteration + 1")
	
	pconv = plot(p1, p2);
end

# ╔═╡ e1a77e7c-574d-47c7-a765-5794e9873e6a
"""
    getcancerdata()

Gets the cancer dataset for binary classification. Returns `features` and
`binary_response`. The feature matrix is standardized, the first column is a
vector of ones as the intercept. The binary responses have true denote Malignant
and false Benign. Internet connection is needed for this function.
"""
function getcancerdata()
    io = HTTP.get("https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/04.Unconstrained/Data/BreastCancer.csv")
    cancer_data = CSV.read(io.body, DataFrame)
    binary_response = cancer_data.y .== "M"
    # extract feature matrix X
    features = Matrix(cancer_data[:,3:end])
    # standarizing features
    # this is needed for gradient descent to run faster
    features .-= sum(features, dims=1) / size(features, 1)
    features ./= sum(features.^2, dims=1).^0.5 / (size(features, 1).^0.5)
    # add intercept using dummy variable
    features = [ones(size(features, 1)) features]
    return features, binary_response
end


# ╔═╡ fdf804b3-28b6-4a04-8ccf-1fa61da264c5
features, binary_response = getcancerdata();

# ╔═╡ 113aef04-5f2a-40ed-a520-10a8420cf4e9
binary_response

# ╔═╡ 7bd0f207-7bb2-4297-a585-cd7002977993
features

# ╔═╡ fdabfe92-a820-459d-8b6e-37d32a7e807b
l_loss(w) = logistic_loss(w, binary_response, features, λ=0.1);

# ╔═╡ 2ddcf834-27d2-4751-ab27-0194fadfe179
l_grad(w) = grad_logistic_loss(w, binary_response, features, λ=0.1);

# ╔═╡ b3e2b862-26be-4fe1-a76b-3704d72cd36c
l_hess(w) = hess_logistic_loss(w, binary_response, features, λ=0.1);

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CSV = "~0.9.4"
DataFrames = "~1.2.2"
HTTP = "~0.9.14"
LaTeXStrings = "~1.2.1"
Plots = "~1.22.1"
PlutoUI = "~0.7.10"
Zygote = "~0.6.22"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "3a877c2fc5c9b88ed7259fd0bdb7691aad6b50dc"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.4"

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
git-tree-sha1 = "bd4afa1fdeec0c8b89dad3c6e92bc6e3b0fec9ce"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.6.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

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

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

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

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

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

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "6d4b609786127030d09e6b1ee0e2044ec20eb403"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.11"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7f6ad1a7f4621b4ab8e554133dade99ebc6e7221"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.5"

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

[[HypertextLiteral]]
git-tree-sha1 = "72053798e1be56026b81d4e2682dbe58922e5ec9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.0"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

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

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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
deps = ["Base64", "Dates", "HypertextLiteral", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "26b4d16873562469a0a1e6ae41d90dec9e51286d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.10"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

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

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ad42c30a6204c74d264692e633133dcea0e8b14e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.2"

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

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

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

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

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

[[WeakRefStrings]]
deps = ["DataAPI", "Parsers"]
git-tree-sha1 = "4a4cfb1ae5f26202db4f0320ac9344b3372136b0"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.3.0"

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
git-tree-sha1 = "4b799addc63aa77ad4112cede8086564d9068511"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.22"

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
# ╟─67e914ba-1b9f-11ec-3109-8b3c4fb77792
# ╠═2a6fa183-5c5a-4d50-a6d6-7a6cef635ddf
# ╟─34f53c61-3315-4eb0-ba96-06fb25e507a2
# ╟─cbdba765-7aa5-4e08-bee1-193e6073112f
# ╟─d39503f3-ada3-4b38-9aec-21650a3d55b3
# ╟─1484214a-5e5e-4a15-8e02-7739e47b18a6
# ╟─73b10099-a769-48af-bb0b-6d3aa785e612
# ╟─7c11ebc0-f2f4-4d50-b691-533e5b996c88
# ╟─c7e1ee9e-6a00-47ca-a540-84029bba301a
# ╟─421c5ff4-d7df-4340-92ab-a38f25a67090
# ╟─3c5b595f-4b63-4036-9d01-7fb1daa60029
# ╟─a48204c9-9bbb-48bc-b1c1-825431b0d0cc
# ╟─34786d52-19e2-4c4b-ac7c-04316a400216
# ╟─06eab55d-a514-424b-9270-a7a0e82f0847
# ╟─0b86ef87-db56-48ce-878f-81d8c65294b5
# ╟─de63922b-584a-4bb9-9d9b-79e75b60cb1d
# ╟─26063cc6-1ac3-4149-9e96-7ffcae53d57d
# ╟─9d4f5f9c-0cbb-4f48-be7f-0365fa8c0cbb
# ╟─64d464af-3403-4ea9-8fd5-983792d7b125
# ╟─fc9bddb6-69e7-44ca-820e-b2b67770067b
# ╠═af3f1a9c-32c6-4e4b-8aeb-feb2cc99160d
# ╠═498c9140-1bd0-4b73-a6f0-57898a80164d
# ╠═286c3b12-cb4d-489a-b4f6-b5a3aafedf9a
# ╠═6fc90c99-4782-4881-bb87-18b0723ff862
# ╟─64a52a2f-65ea-4efe-8f63-826b5cb9745b
# ╟─045e627e-09bf-4589-bada-2d184d68977a
# ╟─723d9b31-eb75-49c1-a33a-9e105d3ccdfb
# ╟─533893ac-b89b-48c0-afa1-2b1284fda494
# ╟─0bca2788-018b-4b54-b910-3d19fed02788
# ╠═33a0c8ac-ba23-4ac2-aafe-c5622b7e2255
# ╠═3f2d01cc-b0e2-4ee1-9e90-eec8a179c447
# ╠═186ee3e3-a36d-440f-bfa1-27efdb80ce1f
# ╟─d3c33ef9-29dc-45e9-aae6-e7bc062971fc
# ╟─d6fcda50-4bd3-49b5-bd2c-11c278e3bb31
# ╟─d6c28b2e-24d0-41dd-931e-586f4a906a90
# ╟─0db2b747-5d0c-4eac-a722-91b8cecf2bd8
# ╠═f38f6ab4-6718-4fe0-bba1-a2fe04592bdd
# ╠═599b2161-f45f-49df-85ca-9a9649ed7f01
# ╠═e95d1fd3-83f9-4001-8c23-01cb45630daa
# ╟─3c3561b2-03d8-46a1-910a-ba5b2b43f082
# ╟─46e6c9a2-6fae-4abc-87f8-d423a63df6ad
# ╟─5015de28-d9f6-4a29-a380-4f059d8b9c66
# ╟─2185cb3e-2a81-43fb-845c-e786ec23d16b
# ╠═a6174578-3c81-4aec-a47d-b48d58e7a120
# ╠═ed7823a9-17c9-4402-a870-528648053bb4
# ╠═c635dade-3c56-4edb-8446-a52e260a66a7
# ╟─3a367342-3742-4c7d-ba2a-6345414bf283
# ╟─5d1634c3-deff-44ae-884d-0d250ce8879c
# ╟─91242142-a86f-4357-b6bd-3496d94ba7a4
# ╟─16aa0de1-1dfc-4168-839f-35b86eca4269
# ╟─07d59dd2-1d0b-4795-a2c4-72d2b522c81b
# ╟─c69f3e77-eaa9-44fa-9274-5d78e559fa1b
# ╟─8467ce85-2d9b-417d-86e4-6f5f6c0c67c2
# ╠═fdf804b3-28b6-4a04-8ccf-1fa61da264c5
# ╠═113aef04-5f2a-40ed-a520-10a8420cf4e9
# ╠═7bd0f207-7bb2-4297-a585-cd7002977993
# ╟─62e9e6ba-b236-48c6-ae17-955444547693
# ╠═851bfa30-1a9b-414c-8584-4e544b84a3ab
# ╠═9abb8ced-16e8-413d-91cd-7f5b108ad044
# ╠═4c93c629-fbbf-4981-ab21-74eab8c29265
# ╠═c994bc37-7e7f-4ca3-ad2a-502df6ff48c5
# ╠═e9135d62-870b-4c31-b772-10f50c5e8059
# ╠═ada77919-3960-4d76-9d72-0a8e343b41dd
# ╠═fdabfe92-a820-459d-8b6e-37d32a7e807b
# ╠═2ddcf834-27d2-4751-ab27-0194fadfe179
# ╠═b3e2b862-26be-4fe1-a76b-3704d72cd36c
# ╟─be59e014-f7ce-4cb3-8cce-bdaa2f1bb2e1
# ╠═5a891bd7-1a9d-40fb-a8fc-9d1fa036e81c
# ╟─f78663c8-1581-48e8-9475-a9a15220e15f
# ╠═75deabf3-3a4b-424d-9231-2ddec5aaf5eb
# ╠═82838573-5cb4-4a4a-83be-af7ec0564b7a
# ╠═f16d253f-aa84-4de9-81ac-f7a8daca6f35
# ╟─1cabff7d-2fd2-4548-94ed-2d3f49496f4b
# ╟─6630572c-47e6-48ac-a2b5-d21f03f317f4
# ╟─86140ad2-6204-4ef4-8644-3c72ba77fbe3
# ╟─17c12a43-b9f9-42b0-886d-92dddf32e767
# ╟─1bcccd24-0f8b-4363-949a-d498d3a41e86
# ╟─e1a77e7c-574d-47c7-a765-5794e9873e6a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
