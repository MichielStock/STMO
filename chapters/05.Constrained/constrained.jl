### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 065de10a-ecf4-413e-ab9c-fafc96e2b822
using Plots, Zygote, LaTeXStrings, PlutoUI

# ╔═╡ 3deb0b7d-7a9f-4ae1-ae50-6d710c3eb391
module Solution
	using LinearAlgebra
	
	export solve_constrained_quadratic_problem, linear_constrained_newton

	"""
    solve_constrained_quadratic_problem(P, q, A, b)

	Solve a linear constrained quadratic convex problem.

	Inputs:
		- P, q: quadratic and linear parameters of
				the linear function to be minimized
		- A, b: system of the linear constraints

	Outputs:
		- xstar: the exact minimizer
		- nustar: the optimal Lagrange multipliers
	"""
	function solve_constrained_quadratic_problem(P, q, A, b)
		p, n = size(A)  # size of the problem
		# complete this code
		solution = [P A'; A zeros(p, p)] \ [-q; b]
		xstar = solution[1:n]
		nustar = solution[(n+1):end]
		return xstar, nustar
	end

	"""
	linear_constrained_newton(f, x₀, ∇f,
				  ∇²f, A, b; t::Real=0.25, ϵ::Real=1e-3,
				  verbose=false)

	Newton's method for minimizing functions with linear constraints.
	Expects a feasible x₀!

	Inputs:
		- f: function to be minimized
		- x₀: starting point (does not have to be feasible)
		- Df: gradient of the function to be minimized
		- DDf: hessian matrix of the function to be minimized
		- A, b: linear constraints
		- t: step size for each Newton step (fixed)
		- ϵ: parameter to determine if the algorithm is converged
		- verbose: print the number of steps?

	Outputs:
		- xstar: the found minimum
	"""
	function linear_constrained_newton(f, x₀, Df,
				  DDf, A, b; t::Real=0.9, ϵ::Real=1e-4, verbose=false)
		@assert 0.0 < t <= 1.0 "stepsize t should be in (0, 1]"
		@assert A * x₀ ≈ b
		x = x₀  # initial value
		p, n = size(A)
		nsteps = 0
		while true
			ddfx = DDf(x)
			dfx = Df(x)
			# calculate residual
			Dx, _ = solve_constrained_quadratic_problem(ddfx, dfx, A, b - A * x) 
			λ² = abs(Dx' * ddfx * Dx)
			if λ²/2 < ϵ  # stopping criterion
				break  # converged
			end
			# perform step
			x .+= t * Dx
			nsteps += 1
		end
		verbose && println("Converged in $nsteps steps")
		return x
	end
end

# ╔═╡ d99b6b70-1bac-11ec-2d2a-775e8436fd8f
md"""
# Constrained convex optimization

*STMO*

**Michiel Stock**

![](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/logo.png?raw=true)
"""

# ╔═╡ b3efad4a-51f0-4fba-9b35-df2e3914a76e
md"""
## Motivation

Many more realistic optimization problems are characterized by constraints. For example, real-world systems often satisfy conservation laws, such as conservation of mass, of atoms or of charge. When designing objects, there are practical constraints of feasible dimensions, range of operations and limitations in materials. Another example is in probability, where a solution should satisfy the axioms of probability theory (probabilities are real values between 0 and 1 and the probabilities of all events should sum to 1).

Other cases, we include constraints in our problem because they encode prior knowledge about the problem or to obtain solutions with certain desirable properties.

In this chapter we discuss convex optimization problems with linear equality constraints (constraining the solution to a linear subspace) and convex inequality constrains (constraining the solution to convex subspace). Both types of constraints result again in a convex optimization problem.
"""

# ╔═╡ 34a331a9-4e2f-41ea-b75b-4dd17d9f0911
md"""
## Lagrange multipliers

Lagrange multipliers are elegant ways of finding stationary points of a function of several variables given one or more constraints. We give a short introduction based on a geometric perspective.

> IMPORTANT: most textbooks treat Lagrange multipliers as maximization problems. Here they are treated as minimization problems to be consistent with other chapters.
"""

# ╔═╡ e6bebf18-4b13-462b-acf4-c02a3855ddbe
md"""
Consider the following optimization problem:

$$\min_{\mathbf{x}} f(\mathbf{x})$$

$$\text{subject to } g(\mathbf{x})=0\,.$$

![Convex optimization problem with an equality constraint. Here, the constraint is nonlinear.](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/Lagr1.png?raw=true)

For every point $\mathbf{x}$ on the surface $g(\mathbf{x})=0$, the gradient $\nabla g(\mathbf{x})$ is normal to this surface. This can be shown by considering a point $\mathbf{x}+\boldsymbol{\epsilon}$, also on the surface. If we make a Taylor expansion around $\mathbf{x}$, we have

$$g(\mathbf{x}+\boldsymbol{\epsilon})\approx g(\mathbf{x}) + \boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})\,.$$

Given that both $\mathbf{x}$ and $\mathbf{x}+\boldsymbol{\epsilon}$ lie on the surface it follows that $g(\mathbf{x}+\boldsymbol{\epsilon})= g(\mathbf{x})$. In the limit that $||\boldsymbol{\epsilon}||\rightarrow 0$ we have that $\boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})=0$. Because $\boldsymbol{\epsilon}$ is parallel to the surface $g(\mathbf{x})$, it follows that $\nabla g(\mathbf{x})$ is normal to the surface.

![The same optimization problem, with some gradients of $f(\mathbf{x})$ and $g(\mathbf{x})$ shown.](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/Lagr2.png?raw=true)

We seek a point $\mathbf{x}^\star$ on the surface such that $f(\mathbf{x})$ is minimized. For such a point, it should hold that the gradient w.r.t. $f$ should be parallel to $\nabla g$. Otherwise, it would be possible to give a small 'nudge' to $\mathbf{x}^\star$ in the direction of $\nabla f$ to decrease the function value, which would indicate that $\mathbf{x}^\star$ is not a minimizer. This figures below illustrate this point.

![Point on the surface that is *not* a minimizer.](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/Lagr3.png?raw=true)

![Point on the surface that is a minimizer of $f$.](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/Lagr4.png?raw=true)

$$\nabla f(\mathbf{x}^\star) + \nu \nabla g (\mathbf{x}^\star)=0\,,$$

with $\nu\neq 0$ called the *Lagrange multiplier*. The constrained minimization problem can also be represented by a *Lagrangian*:

$$L(\mathbf{x}, \nu) 	\equiv f(\mathbf{x}) + \nu g(\mathbf{x})\,.$$

The constrained stationary condition is obtained by setting $\nabla_\mathbf{x} L(\mathbf{x}, \nu) =0$, the condition $\partial  L(\mathbf{x}, \nu)/\partial \nu=0$ leads to the constraint equation $g(\mathbf{x})=0$.
"""

# ╔═╡ e2b613a4-b9e1-42d6-a55d-096a8d6e9c68
md"""
### Inequality constraints

The same argument can be made for inequality constraints, i.e. solving

$$\min_{\mathbf{x}} f(\mathbf{x})$$

$$\text{subject to } g(\mathbf{x})\leq0\,.$$

Here, two situations can arise:

- **Inactive constraint**: the minimizer of $f$ lies in the region where $g(\mathbf{x}) < 0$. This corresponds to a Lagrange multiplier $\nu=0$. Note that the solution would be the same if the constraint was not present.
- **Active constraint**: the minimizer of $f$ lies in the region where $g(\mathbf{x}) > 0$. The solution of the constrained problem will lie on the bound where $g(\mathbf{x})=0$, similar to the equality-constrained problem and corresponds to a Lagrange multiplier $\nu>0$.

Both scenarios are shown below:

![Constrained minimization problem with an active inequality constraint. Optimum lies within the region where $g(\mathbf{x})\leq 0$. ](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/Lagr6.png?raw=true)

![Constrained minimization problem with an active inequality constraint. Optimum lies on the boundary of the region where $g(\mathbf{x})\leq 0$.](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/Lagr5.png?raw=true)


For both cases, the product $\nu g(\mathbf{x})=0$, the solution should thus satisfy the following conditions:

$$g(\mathbf{x}) \leq 0$$

$$\nu \geq 0$$

$$\nu g(\mathbf{x})=0\,.$$

These are called the *Karush-Kuhn-Tucker* conditions.

It is relatively straightforward to extend this framework towards multiple constraints (equality and inequality) by using several Lagrange multipliers.
"""

# ╔═╡ c35a1d9a-dd58-4d31-83fc-3d0fe9562078
md"""
## Equality constrained convex optimization

### Problem outline

We will start with convex optimization problems with linear equality constraints:

$$\min_\mathbf{x} f(\mathbf{x})$$

$$\text{subject to } A\mathbf{x}=\mathbf{b}$$

where $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is convex and twice continuously differentiable and $A\in \mathbb{R}^{p\times n}$ with a rank $p < n$.

The Lagrangian of this problem is

$$L(\mathbf{x}, \boldsymbol{\nu}) = f(\mathbf{x}) + \boldsymbol{\nu}^\top(A\mathbf{x}-\mathbf{b})\,,$$

with $\boldsymbol{\nu}\in\mathbb{R}^p$ the vector of Lagrange multipliers.

A point $\mathbf{x}^\star\in$ **dom** $f$ is optimal for the above optimization problem only if there is a $\boldsymbol{\nu}^\star\in\mathbb{R}^p$ such that:

$$A\mathbf{x}^\star = \mathbf{b}, \qquad \nabla f(\mathbf{x}^\star) + A^\top\boldsymbol{\nu}^\star = 0\,.$$
"""

# ╔═╡ 43226611-b76d-42e6-9b1e-8f6d88ce4b86
md"""
We will reuse the same toy examples from the previous chapter, but add an equality constraint to both.

- Simple quadratic problem:

$$\min_{\mathbf{x}} \frac{1}{2} (x_1^2 + 4 x_2^2)$$

$$\text{subject to }  x_1 - 2x_2 = 3$$

- A non-quadratic function:

$$\min_{\mathbf{x}}\log(e^{x_1 +3x_2-0.1}+e^{x_1 -3x_2-0.1}+e^{-x_1 -0.1})$$

$$\text{subject to }  x_1 + 3x_2 = 0$$
"""

# ╔═╡ 37a4e294-8807-47ab-ab08-c3d45442e2f9
md"""
### Equality constrained convex quadratic optimization

Consider the following equality constrained convex optimization problem:

$$\min_\mathbf{x}\frac{1}{2}\mathbf{x}^\top P \mathbf{x} + \mathbf{q}^\top \mathbf{x} + r$$

$$\text{subject to }  A\mathbf{x}=\mathbf{b}$$

where $P$ is symmetric.

The optimality conditions are

$$A\mathbf{x}^\star = \mathbf{b}, \quad P\mathbf{x}^\star+\mathbf{q} +A^\top\boldsymbol{\nu}^\star=\mathbf{0}\,,$$

which we can write as

$$\begin{bmatrix}
P & A^\top \\
A & 0 \\
     \end{bmatrix}
     \begin{bmatrix}
\mathbf{x}^\star\\
\boldsymbol{\nu}^\star
     \end{bmatrix}
     =
     \begin{bmatrix}
-\mathbf{q} \\
\mathbf{b}
     \end{bmatrix}\,.$$

Note that this is a block matrix.

> If $P$ is positive-definite, the linearly constrained quadratic minimization problem has an unique solution.

Solving this linear system gives both the constrained minimizer $\mathbf{x}^\star$ as well as the Lagrange multipliers.
"""

# ╔═╡ f9e182e1-16ac-4ff6-b6b5-8b1e0e84c1b8
md"**Assignment 1**

1. Complete the code to solve linearly constrained quadratic systems.
2. Use this code to solve the quadratic toy problem defined above."

# ╔═╡ d9453447-c33d-4705-98b2-35e4ad945af0
"""
    solve_constrained_quadratic_problem(P, q, A, b)

Solve a linear constrained quadratic convex problem.

Inputs:
    - P, q: quadratic and linear parameters of
            the linear function to be minimized
    - A, b: system of the linear constraints

Outputs:
    - xstar: the exact minimizer
    - nustar: the optimal Lagrange multipliers
"""
function solve_constrained_quadratic_problem(P, q, A, b)
    p, n = missing  # size of the problem
    # complete this code
    solution = missing
    xstar = missing
    nustar = missing
    return xstar, nustar
end

# ╔═╡ 0b07677f-0e1c-4787-b4ca-e35cd67f0ad1
 # solve the quadratic system with the linear constraint

# ╔═╡ fd1ddac1-53b7-48b4-adf4-4f196db5d1e1


# ╔═╡ 080ba202-520d-4acc-aac0-41f70aa08117
md"""
### Newton's method with equality constraints

To derive $\Delta \mathbf{x}_{nt}$ for the following equality constrained problem

$$\min_\mathbf{x}  f(\mathbf{x})$$

$$\text{subject to }  A\mathbf{x}=\mathbf{b}$$

we apply a second-order Taylor approximation at the point $\mathbf{x}$, to obtain

$$\min_\mathbf{v} \hat{f}(\mathbf{x} +\mathbf{v}) = f(\mathbf{x}) +\nabla f(\mathbf{x})^\top \mathbf{v}+ \frac{1}{2}\mathbf{v}^\top \nabla^2 f(\mathbf{x}) \mathbf{v}$$

$$\text{subject to } A(\mathbf{x}+\mathbf{v})=\mathbf{b}\,.$$

Based on the solution of quadratic convex problems with linear constraints, the Newton $\Delta \mathbf{x}_{nt}$ step is characterized by

$$\begin{bmatrix}
 \nabla^2 f(\mathbf{x})&  A^\top \\
A & 0 \\
     \end{bmatrix}
     \begin{bmatrix}
\Delta \mathbf{x}_{nt}\\
\mathbf{w}
     \end{bmatrix}
     =
     -\begin{bmatrix}
\nabla f(\mathbf{x}) \\
A\mathbf{x}-\mathbf{b}
     \end{bmatrix}$$

- If the starting point $\mathbf{x}^{(0)}$ is chosen such that $A\mathbf{x}^{(0)}=\mathbf{b}$, the residual term vanishes and steps will remain in the feasible region. This is the **feasible start Newton method**.
- If we choose an arbitrary $\mathbf{x}^{(0)}\in$ **dom** $f$, not satisfying the constraints, this is the **infeasible start Newton method**. It will usually converge rapidly to the feasible region (check the final solution!).

Note that when we start at a feasible point, the residual vector $-(A\mathbf{x}-\mathbf{b})$ vanishes and the path will always remain in a feasible region. Otherwise we will converge to it.

In this chapter, we will use a fixed step size. For Newton's method this usually leads to only a few extra iterations compared to an adaptive step size.

>**input** starting point $\mathbf{x}\in$ **dom** $f$ (with $A\mathbf{x}=\mathbf{b}$ if using the feasible method), tolerance $\epsilon>0$.
>
>**repeat**
>
>>    1. Compute the Newton step $\Delta \mathbf{x}_{nt}$ and decrement $\lambda(\mathbf{x})$.
>>    2. *Stopping criterion*. **break** if $\lambda^2/2\leq \epsilon$.
>>    3. *Choose step size $t$*: either by line search or fixed $t$.
>>    4. *Update*. $\mathbf{x}:=\mathbf{x}+t \Delta \mathbf{x}_{nt}$.
>
>**output** $\mathbf{x}$

Again, the convergence can be monitored using the Newton decrement:

$$\lambda^2(\mathbf{x}) = \Delta \mathbf{x}_{nt}^\top \nabla^2 f(\mathbf{x})\Delta \mathbf{x}_{nt}\,.$$

The algorithm terminates when

$$\frac{\lambda(\mathbf{x})^2}{2} < \epsilon\,.$$

The Newton decrement also indicates whether we are in or close to the feasible region.
"""

# ╔═╡ 238a0d97-aeca-4a0b-9458-ec223281bbec
md"**Assignment 2**

1. Complete the code for the linearly constrained Newton method.
2. Use this code to find the minimum of the non-quadratic toy problem, defined above (compare a feasible and infeasible start).
"

# ╔═╡ e712c1b5-fe77-457f-8816-4db8f6c34699
"""
linear_constrained_newton(f, x₀, ∇f,
              ∇²f, A, b; t::Real=0.25, ϵ::Real=1e-3,
              verbose=false)

Newton's method for minimizing functions with linear constraints.
Expects a feasible x₀!

Inputs:
    - f: function to be minimized
    - x₀: starting point (does not have to be feasible)
    - Df: gradient of the function to be minimized
    - DDf: hessian matrix of the function to be minimized
    - A, b: linear constraints
    - t: step size for each Newton step (fixed)
    - ϵ: parameter to determine if the algorithm is converged
    - verbose: print the number of steps?

Outputs:
    - xstar: the found minimum
"""
function linear_constrained_newton(f, x₀, ∇f,
              ∇²f, A, b; t::Real=0.25, ϵ::Real=1e-4, verbose=false)
    @assert 0.0 < t <= 1.0 "stepsize t should be in (0, 1]"
    @assert missing # test if x₀ is feasible
    x = x₀  # initial value
    p, n = size(A)
    nsteps = 0
    while true
        dfx = ∇f(x)
        ddfx = ∇²f(x)
        # calculate residual
        Dx, _ = missing # complete!
        λ² = missing
        if missing  # stopping criterion
            break  # converged
        end
        # perform step
        missing
        nsteps += 1
    end
    verbose && println("Converged in $nsteps steps")
    return x
end

# ╔═╡ 2fc56050-6be3-46a4-be2b-a8402fe15b23
 # solve the nonlinear system

# ╔═╡ dbc38fb2-4ce7-427b-87d0-d5059ae10027
md"""
## Inequality constrained convex optimization

### Inequality constrained minimization problems

$$\min_\mathbf{x}  f_0(\mathbf{x})$$

$$\text{subject to } f_i(\mathbf{x}) \leq 0, \quad i=1,\ldots,m$$

$$A\mathbf{x}=\mathbf{b}$$

where $f_0,\ldots,f_m\ :\ \mathbb{R}^n \rightarrow \mathbb{R}$ are convex and twice continuously differentiable, and $A\in \mathbb{R}^{p\times n}$ with **rank** $A=p<n$.

**Question 1**

Write the inequality constraints for the case when:
1. either $x_k\geq 0$
2. or $\mathbf{x}$ lies in the unit sphere.

Using the theory of Lagrange multipliers, a point $\mathbf{x}^\star \in\mathbb{R}^n$ is optimal if and only if there exist a $\boldsymbol{\lambda}^\star\in \mathbb{R}^m$ and $\boldsymbol{\nu}^\star\in \mathbb{R}^p$ such that

$$A\mathbf{x}^\star=\mathbf{b}$$

$$f_i(\mathbf{x}^\star) \leq 0, \quad i=1,\ldots,m$$

and

$$\lambda_i^\star \geq 0, \quad i=1,\ldots,m$$

$$\nabla f_0(\mathbf{x}^\star)+\sum_{i=1}^m\lambda^\star_i\nabla f_i(\mathbf{x}^\star) +A^\top \boldsymbol{\nu}^\star=0$$

$$\lambda_if_i(\mathbf{x}^\star)=0, \quad i=1,\ldots,m\,.$$
"""

# ╔═╡ 1235659a-50a4-47b6-ad8e-cd20c8888a6e
md"""
### Implicit constraints

Rather than solving a minimization problem with inequality constraints, we can reformulate the objective function to include only the feasible regions:

$$\min_{\mathbf{x}} f_0(\mathbf{x})+\sum_{i=1}^m I_{-}(f_i(\mathbf{x}))$$

$$A\mathbf{x}=\mathbf{b}$$

where $I_{-}:\mathbb{R}\rightarrow \mathbb{R}$ is the *indicator function* for the nonpositive reals:

$$I_-(u) = 0 \text{ if } u\leq 0$$

and

$$I_-(u) = \infty \text{ if } u> 0\,.$$

Sadly, we cannot directly optimize such a function using gradient-based optimization as $I_-$ does not provide gradients to guide us.
"""

# ╔═╡ 761a847c-e54f-462f-a0b1-acf99f013865
md"""
**Example**

The non-quadratic function with inequality constraints:

$$\min_\mathbf{x} f_0(x_1, x_2)   = \log(e^{x_1 +3x_2-0.1}+e^{x_1 -3x_2-0.1}+e^{-x_1 -0.1})$$

$$\text{subject to }  (x_1 - 1)^2 + (x_2 - 0.25)^2 \leq 1$$

![Convex function with an equality constraint. Note that the feasible region is a convex set.](https://github.com/MichielStock/STMO/blob/master/chapters/05.Constrained/Figures/ineq_const_example.png?raw=true)
"""

# ╔═╡ c11d17ac-c741-4851-b430-d990f6b5ada2
md"""
### Logarithmic barrier

Main idea: approximate $I_-$ by the function:

$$\hat{I}_-(u) = - (1/t)\log(-u) \text{ if } u< 0$$

and

$$\hat{I}_-(u)=\infty  \text{ if } u\geq 0$$
where $t>0$ is a parameter that sets the accuracy of the approximation.

Thus the problem can be approximated by:

$$\min_\mathbf{x} f_0(\mathbf{x}) +\sum_{i=1}^m\hat{I}_-(f_i(\mathbf{x}))$$

$$\text{subject to } A\mathbf{x}=\mathbf{b}\,.$$

Note that:

- since $\hat{I}_-(u)$ is convex and  increasing in $u$, the objective is also convex;
- unlike the function $I$, the function $\hat{I}_-(u)$ is differentiable;
- as $t$ increases, the approximation becomes more accurate, as shown below.
"""

# ╔═╡ c1ca0e35-346d-49b5-9ded-c7f820f5377e
@bind logt Slider(-5:0.2:4, default=0)

# ╔═╡ f8ec235e-617d-435a-aea4-e70e98fc5e68
t = exp(logt)

# ╔═╡ f20056c4-b635-4df4-b9a2-97b5112bcf7b
Î(u; t) = u < 0 ? - (1/t) * log(-u) : Inf

# ╔═╡ f768e126-2dc2-410c-afba-ea4819384566
md"""
### The barrier method

The function

$$\phi (\mathbf{x}) =\sum_{i=1}^m-\log(-f_i(\mathbf{x}))\,$$

is called the *logarithmic barrier* for the constrained optimization problem.

The new optimization problem becomes:

$$\min_\mathbf{x} tf_0(\mathbf{x}) +\phi (\mathbf{x})$$

$$\text{subject to } A\mathbf{x}=\mathbf{b}\,.$$

- The parameter $t$ determines the quality of the approximation, the higher the value the closer the approximation matches the original problem.
- The drawback of higher values of $t$ is that the problem becomes harder to optimize using Newton's method, as its Hessian will vary rapidly near the boundary of the feasible set.
- This can be circumvented by solving a sequence of problems with increasing $t$ at each step, starting each Newton minimization at the solution of the previous value of $t$.

Computed for you:

- gradient of $\phi$:
$$\nabla\phi(\mathbf{x}) = \sum_{i=1}^m\frac{1}{-f_i(\mathbf{x})} \nabla f_i(\mathbf{x})$$
- Hessian of $\phi$:
$$\nabla^2\phi(\mathbf{x}) = \sum_{i=1}^m \frac{1}{f_i(\mathbf{x})^2} \nabla f_i(\mathbf{x}) \nabla f_i(\mathbf{x})^\top+\sum_{i=1}^m\frac{1}{-f_i(\mathbf{x})^2} \nabla^2 f_i(\mathbf{x})$$
"""

# ╔═╡ 3d4de6a1-73c9-4b79-9111-4a052af7637c
md"The pseudocode of the **barrier method** is given below. We start with a low value of $t$ and increase every step with a factor $\mu$ until $m/t$ is smaller than some $\epsilon>0$.


>**input** strictly feasible $\mathbf{x}$, $t:=t^{(0)}>0, \mu>1$, tolerance $\epsilon>0$.
>
>**repeat**
>
>>    1. *Centering step*.<br>
>>   Compute $\mathbf{x}^\star(t)$ by minimizing $tf_0+\phi$, subject to $A\mathbf{x}=\mathbf{b}$, starting at $\mathbf{x}$.
>>    2. *Update*. $\mathbf{x}:=\mathbf{x}^\star(t)$
>>    3. *Stopping criterion*. **quit** if $m/t<\epsilon$.
>>    4. *Increase $t$.*  $t:=\mu t$.
>
>**until** $m/t < \epsilon$
>
>**output** $\mathbf{x}$
"

# ╔═╡ ab88b4b9-913c-4e4d-af78-bcc162820f2c
md"**Choice of $\mu$**

The choice has a trade-off in the number of inner and outer iterations required:
- If $\mu$ is small (close to 1) then $t$ increases slowly. A large number of Newton iterations will be required, but each will go fast.
- If $\mu$ is large then $t$ increases very fast. Each Newton step will take a long time to converge, but few iterations will be needed.

The exact value of $\mu$ is not particularly critical, values between 10 and 20 work well.

**Choice of $t^{(0)}$**

- If $t^{(0)}$ is chosen too large: the first outer iteration will require many iterations.
- If $t^{(0)}$ is chosen too small: the algorithm will require extra outer iterations.
"

# ╔═╡ a4a629d2-fd57-424b-9538-519133bcb3e8
md"### Central path

The *central path* is the set of points satisfying:

-  $\mathbf{x}^\star(t)$ is strictly feasible: $A\mathbf{x}^\star(t)=\mathbf{b}$ and $f_i(\mathbf{x}^\star(t))<0$ for $i=1,\ldots,m$
- there exist a $\hat{\boldsymbol{\nu}}\in\mathbb{R}^p$ such that
$$t\nabla f_0(\mathbf{x}^\star(t)) + \nabla \phi(\mathbf{x}^\star(t)) +A^\top \hat{\boldsymbol{\nu}}=0$$
- one can show that $f_0(\mathbf{x}^\star(t))-p^\star\leq m / t$: $f_0(\mathbf{x}^\star(t))$ converges to an optimal point as $t\rightarrow \infty$.
"

# ╔═╡ 8d80517f-1c4b-4586-a21b-85a07620fc6f
md"## References

- Boyd, S. and Vandenberghe, L., '*[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)*'. Cambridge University Press (2004)
- Bishop, C., *Pattern Recognition and Machine Learning*. Springer (2006)
"

# ╔═╡ ea1942ea-ea60-4898-a646-6dfe43be6454
begin
	myblue = "#304da5"
	mygreen = "#2a9d8f"
	myyellow = "#e9c46a"
	myorange = "#f4a261"
	myred = "#e76f51"
	myblack = "#50514F"

	mycolors = [myblue, myred, mygreen, myorange, myyellow]
end;

# ╔═╡ ac5128a2-b96b-4e5e-b121-cebbc8010bae
begin
	plot([-3, 0, 0], [0, 0, 10], ls=:dash, color=myblue, label="\$I(u)\$", lw=2)
	plot!(u->Î(u;t), -3, -1e-10, color=myred, lw=2, label="\$\\hat{I}(u)\$")
	xlims!((-3, 1))
	ylims!((-3, 11))
end

# ╔═╡ 008cc76a-c326-4cb7-a166-080ca2c8eb83
begin
	
	fquadr((x1, x2); γ=10.0) = 0.5(x1^2 + γ * x2^2);
	grad_fquadr((x1, x2); γ=10.0) = [x1, γ * x2];
	hess_fquadr((x1, x2); γ=10.0) = [1 0; 0 γ];
	
	fnonquadr((x1, x2)) = log(exp(x1+3x2-0.1) + exp(x1-3x2-0.1)+exp(-x1-0.1));
	grad_fnonquadr(x) = Zygote.gradient(fnonquadr, x)[1];
	hess_fnonquadr(x) = Zygote.hessian(fnonquadr, x);

end

# ╔═╡ 5dabeee0-4ba3-4411-b9cd-fa9a4220eabe
begin
	pq = contourf(-10:0.1:10, -5:0.1:5, (x1, x2) -> fquadr((x1, x2)), xlabel="\$ x_1 \$", color=:blues,
                ylabel="\$ x_2 \$", title="quadratic")
	plot!(x1->(x1-3)/2, -7, 10, lw=2, color=mygreen, label="constraint")
	pq
end

# ╔═╡ ee40098d-9530-43db-b1ca-f9be21b30a3c
fquadr([2, 4]), grad_fquadr([2, 4]), hess_fquadr([2, 4])

# ╔═╡ d35541ab-7639-4afb-b738-e973c0424208
begin
	pnq = contourf(-2:0.1:2, -1:0.1:1, (x1, x2) -> fnonquadr((x1, x2)), xlabel="\$ x_1 \$",
                ylabel="\$ x_2 \$", title="non-quadratic", color=:blues)
	plot!(x1->-x1/3, -2, 2, lw=2, color=mygreen, label="constraint")
	pnq
end

# ╔═╡ dbef3f80-1875-48bc-b702-38de4e1d1631
fnonquadr([2, 4]), grad_fnonquadr([2, 4]), hess_fnonquadr([2, 4])

# ╔═╡ 3116eecd-9d66-412f-8c01-1b94bec094e5


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
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

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

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

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

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

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

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
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

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
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

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
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

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
# ╟─d99b6b70-1bac-11ec-2d2a-775e8436fd8f
# ╠═065de10a-ecf4-413e-ab9c-fafc96e2b822
# ╟─b3efad4a-51f0-4fba-9b35-df2e3914a76e
# ╟─34a331a9-4e2f-41ea-b75b-4dd17d9f0911
# ╟─e6bebf18-4b13-462b-acf4-c02a3855ddbe
# ╟─e2b613a4-b9e1-42d6-a55d-096a8d6e9c68
# ╟─c35a1d9a-dd58-4d31-83fc-3d0fe9562078
# ╟─43226611-b76d-42e6-9b1e-8f6d88ce4b86
# ╟─5dabeee0-4ba3-4411-b9cd-fa9a4220eabe
# ╠═ee40098d-9530-43db-b1ca-f9be21b30a3c
# ╟─d35541ab-7639-4afb-b738-e973c0424208
# ╠═dbef3f80-1875-48bc-b702-38de4e1d1631
# ╟─37a4e294-8807-47ab-ab08-c3d45442e2f9
# ╟─f9e182e1-16ac-4ff6-b6b5-8b1e0e84c1b8
# ╠═d9453447-c33d-4705-98b2-35e4ad945af0
# ╠═0b07677f-0e1c-4787-b4ca-e35cd67f0ad1
# ╠═fd1ddac1-53b7-48b4-adf4-4f196db5d1e1
# ╟─080ba202-520d-4acc-aac0-41f70aa08117
# ╟─238a0d97-aeca-4a0b-9458-ec223281bbec
# ╠═e712c1b5-fe77-457f-8816-4db8f6c34699
# ╠═2fc56050-6be3-46a4-be2b-a8402fe15b23
# ╟─dbc38fb2-4ce7-427b-87d0-d5059ae10027
# ╟─1235659a-50a4-47b6-ad8e-cd20c8888a6e
# ╟─761a847c-e54f-462f-a0b1-acf99f013865
# ╟─c11d17ac-c741-4851-b430-d990f6b5ada2
# ╟─c1ca0e35-346d-49b5-9ded-c7f820f5377e
# ╟─f8ec235e-617d-435a-aea4-e70e98fc5e68
# ╠═f20056c4-b635-4df4-b9a2-97b5112bcf7b
# ╟─ac5128a2-b96b-4e5e-b121-cebbc8010bae
# ╟─f768e126-2dc2-410c-afba-ea4819384566
# ╟─3d4de6a1-73c9-4b79-9111-4a052af7637c
# ╟─ab88b4b9-913c-4e4d-af78-bcc162820f2c
# ╟─a4a629d2-fd57-424b-9538-519133bcb3e8
# ╟─8d80517f-1c4b-4586-a21b-85a07620fc6f
# ╟─ea1942ea-ea60-4898-a646-6dfe43be6454
# ╟─008cc76a-c326-4cb7-a166-080ca2c8eb83
# ╟─3deb0b7d-7a9f-4ae1-ae50-6d710c3eb391
# ╠═3116eecd-9d66-412f-8c01-1b94bec094e5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
