### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 3ea4f8be-d0a7-455d-b58e-3e81ea61f9c1
using Plots,Images, Combinatorics, PlutoUI, Colors, ImageIO, LinearAlgebra

# ╔═╡ 8de53f65-0d4a-4af0-a890-609ddcc92b36
module Solution
	function monge_brute_force(C)
	  n, m = size(C)
	  @assert n == m "C should be square"
	  # loop over all permutations and to find the
	  # matching with the lowest cost

	  best_cost = typemax(eltype(C))
	  for p in permutations(1:n)
		  cost = zero(eltype(C))
		  for (i, j) in enumerate(p)
			cost += C[i,j]
		  end
		  if cost < best_cost
			best_cost = cost
			global best_perm = p
		  end
		end
	  return best_perm, best_cost
	end

	export X1, X2, monge_brute_force


	# SINKHORN
	# -------
    mean(x) = sum(x) / length(x)

	using LinearAlgebra

	function sinkhorn(C::Matrix, a::Vector, b::Vector; λ=1.0, ϵ=1e-8)
		n, m = size(C)
		@assert n == length(a) && m == length(b) throw(DimensionMismatch("a and b do not match"))
		@assert sum(a) ≈ sum(b) "a and b don't have equal sums"
		u, v = copy(a), copy(b)
		M = exp.(-λ * (C .- mean(C)))
		# normalize this matrix
		while maximum(abs.(a .- Diagonal(u) * (M * v))) > ϵ
			u .= a ./ (M * v)
			v .= b ./ (M' * u)
		  end
		return Diagonal(u) * M * Diagonal(v)
	  end

	export sinkhorn
	
end

# ╔═╡ 6f9362d0-1bb6-11ec-2b8b-491ef1f091e5
md"""
# Optimal Transportation

*STMO*

**Michiel Stock**

![](https://github.com/MichielStock/STMO/blob/master/chapters/06.OptimalTransport/Figures/logo.png?raw=true)
"""

# ╔═╡ ccb9bf05-1830-4c56-bf28-312c5d324484
md"""
## Motivation

Optimal transportation deals with transforming one distribution into another while minimizing a cost function. The problem can be formulated as a maximum entropy function, leading to an elegant solution using the *Sinkhorn algorithm*. Optimal transportation is used in computer vision, machine learning, astonomy, bioinformatics, ecology etc.
"""

# ╔═╡ 65a73b72-967d-4a62-af17-f71b286f3303
md"""
## Motivating example: a party in the research group

Let's have a party in our research unit! Pastries and party hats for everyone! We ask Tinne, our laboratory manager, to make some desserts: an airy merveilleux, some delicious eclairs, a big bowl of dark chocolate mousse, a sweet passion fruit-flavored bavarois and moist carrot cake (we got to have our vegetables). If we mentally cut all these sweets into portions, we have twenty portions.
"""

# ╔═╡ ed3f476c-2205-43a7-a4cf-5941a0d0afd0
staff = ["Bernard", "Jan", "Willem", "Hilde", "Steffie", "Marlies", "Tim", "Wouter"]

# ╔═╡ 0e579d2e-9b78-4a56-a30d-883cec0ad5d6
desserts = ["merveilleux", "eclair", "chocolate mousse", "bavarois", "carrot cake"]

# ╔═╡ ce639d43-7481-43f4-8724-1da9c18d716a
a = [3.0, 3, 3, 4, 2, 2, 2, 1]

# ╔═╡ c32441ee-c62d-4ee7-8e1b-ff7de1072d43
b = [4.0, 2, 6, 4, 4]

# ╔═╡ 245acb05-d58b-4ca6-9cf4-e59ff50e86c9
md"As engineers and mathematicians, we pride ourselves in doing things the optimal way. So how can we divide the desserts to make everybody as happy as possible? I went around and asked which of those treats they liked. On a scale between -2 and 2, with -2 being something they hated and 2 being their absolute favorite, the desert preferences of the teaching staff is given below (students: take note!)."

# ╔═╡ 114c8c91-34f3-4256-ab0f-c1fbd64e4827
preferences = [2 2 1 0 0;
              0 -2 -2 -2 2;
              1 2 2 2 -1;
              2 1 0 1 -1;
              0.5 2 2 1 0;
              0 1 1 1 -1;
             -2 2 2 1 1;
              2 1 2 1 -1]

# ╔═╡ 01e3b967-e441-4041-8bd8-5cfc0e0ebaa4
Cdessert = -preferences;  # cost

# ╔═╡ 7dbf484a-f3fb-4510-b500-57f4b13c0f59
begin
	heatmap(preferences, title="preferences", color=:balance)
	yticks!(1:8, staff)
	xticks!(1:5, desserts)
end

# ╔═╡ b3440682-8443-4052-bbf2-8f593c4e54c3
md"See how most people like eclairs and chocolate mousse, but merveilleus are a more polarizing dessert! Jan is lactose intolerant, so he only gave a high score to the carrot cake by default.

The task is clear: divide these desserts in such a way that people get their portions of the kinds they like the most!
"

# ╔═╡ 984effce-9a76-4bbb-9105-06887dfa629e
md"""
## The Monge problem

The original version of optimal transport was formulated by Gaspard Monge in 1781. Here, $n=m$ and the goal is to connect $n$ sources with $n$ sinks to minimize a cost:

$$\min_{\sigma\in\text{Perm(n)}} \frac{1}{n}\sum_{i=1}^nC_{i,\sigma(i)}\,,$$

with $\text{Perm(n)}$ the set of all permutation of $n$ elements.

> **Example**: There are $n$ mines mining iron ore and a collection of $n$ factories. Given a distance between every mine and every factory, select one factory for every mine such that the total cost (=transportation distance) is minimized.

The **Monge problem** is a *discrete combinatorial optimization problem*:

- The size of the search space is $n!$, for $n=70$, there are more than $10^{100}$ permutations!
- Can be solved using the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm).
- Restrictive: two sets to match must be of same size. How to deal with different weights?

For working with probability distributions, we require a way to perform *soft-matching*.
"""

# ╔═╡ 9567f11f-486a-4960-8b69-42d2e5ee413f
md"""
### Exercise: cell tracking

In a microscopy imaging experiment we monitor ten moving cells at time $t_1$ and some time later at time $t_2$. Between these times, the cells have moved. An image processing algorithm determined the coordinates of every cell in the two images. We want to know which cell in the first corresponds to the second image. To this end, search the assignment that minimizes the sum of the squared Euclidian distances between cells from the first image versus the corresponding cell of the second image.

1. `X1` and `X2` contain the $x,y$ coordinates of the cells for the two images. Compute the matrix $C$ containing the pairwise squared Euclidean distance. You can use the function `dist` provided by us.
2. Complete the function `monge_brute_force` to use brute-force search for the best permutation. You might find the function `permutations` from the `Combinatorics` library useful. How large is your search space? Time your function.
3. Make a plot connecting the cells.
"""

# ╔═╡ b85b69fb-bfdb-41bd-a6a1-1aebe3077e08
with_terminal() do
	for p in permutations([1, 2, 3])
  		println(p)
	end
end

# ╔═╡ cc62b7b1-fe34-4a19-846d-ca6c7fb69ff9
function monge_brute_force(C)
   n, m = size(C)
   @assert n == m "C should be square"
   best_perm = nothing
   best_cost = Inf
   for p in permutations(1:n)
		cost = 0.0
		for (i, pi) in enumerate(p)
			cost += C[i,pi]
		end
		if cost < best_cost
			best_cost = cost
			best_perm = p
		end
	end
  
   return best_perm, best_cost
end

# ╔═╡ 4bf0f9d1-806c-40e2-8ef4-325be3a5b2ca
md"one liner (bragging)"

# ╔═╡ 6285291e-0520-40f1-b39e-a98991995e47
monge_brute_force_oneline(C) = minimum(((sum(I->C[I...], enumerate(p)), p)
								for p in permutations(1:size(C,1)))) |> reverse

# ╔═╡ da2cc178-64c7-4b30-909a-24fb402a489b
md"""
## The optimal transportation problem

Let us introduce some notation so we can formally state this as an optimization problem. Let $\mathbf{a}$ be the vector containing the amount of dessert every person may eat. In this case $\mathbf{a} = [3,3,3,4,2,2,2,1]^\intercal$ (in general the dimension of $\mathbf{r}$ is $n$). Similarly, $\mathbf{b}$ denotes the vector of how much there is of every dessert, i.e. $\mathbf{b}=[4, 2, 6, 4, 4]^\intercal$ (in general the dimension of $\mathbf{b}$ is $m$). Often $\mathbf{a}$ and $\mathbf{b}$ represent marginal probability distributions, hence their values are nonzero and sum to one.

Let $U(\mathbf{a}, \mathbf{b})$ be the set of positive $n\times m$ matrices for which the rows sum to $\mathbf{a}$ and the columns sum to $\mathbf{b}$:

$$U(\mathbf{a}, \mathbf{b}) = \{P\in \mathbb{R}_{>0}^{n\times m}\mid P\mathbf{1}_m = \mathbf{a}, P^\intercal\mathbf{1}_n = \mathbf{b}\}\,.$$

For our problem, $U(\mathbf{a}, \mathbf{b})$ contains all the ways of dividing the desserts for my colleagues. Note that we assume here that we can slice every dessert however we like. We do not have only to give whole pieces of pie but can provide any fraction we want.

The preferences of each person for each dessert is also stored in a matrix. To be consistent with the literature, this will be stored in a $n\times m$ *cost* matrix $C$. The above matrix is a preference matrix that can easily be changed into a cost matrix by flipping the sign of every element.

So finally, the problem we want to solve is formally posed as

$$d_C(\mathbf{r}, \mathbf{c}) = \min_{P\in U(\mathbf{a}, \mathbf{b})}\, \sum_{i,j}P_{ij}C_{ij}\,.$$

This is called the *optimal transportation* between $\mathbf{a}$ and $\mathbf{b}$. It can be solved relatively easily using linear programming.

The optimum, $d_C(\mathbf{a}, \mathbf{b})$, is called the *Wasserstein metric*. It is basically a distance between two probability distributions. It is sometimes also called the *earth mover distance* as it can be interpreted as how much 'dirt' you have to move to change one 'landscape' (distribution) in another.
"""

# ╔═╡ bdd25212-576e-4b81-8178-1212ac759cbc
md"""
## Choosing a bit of everything

Consider a slightly modified form of the optimal transport:

$$d_C^\lambda(\mathbf{a}, \mathbf{b}) = \min_{P\in U(\mathbf{a}, \mathbf{b})}\, \sum_{i,j}P_{ij}C_{ij} - \frac{1}{\lambda}H(P)\,,$$

in which the minimizer $d^\lambda_C(\mathbf{a}, \mathbf{b})$ is called the *Sinkhorn distance*. Here, the extra term

$$H(P) = -\sum_{i,j}P_{ij}\log P_{ij}$$

is the *information entropy* of $P$. One can increase the entropy by making the distribution more homogeneous, i.e., giving everybody a more equal share of every dessert. The parameter $\lambda$ determines the trade-off between the two terms: trying to give every person only their favorites or encouraging equal distributions. Machine learners will recognize this as similar to regularization in, for example, ridge regression. Similar to that for machine learning problems, a tiny bit of shrinkage of the parameter can lead to improved performance, the Sinkhorn distance is also observed to work better than the Wasserstein distance on some problems. This is because we use a very natural prior on the distribution matrix $P$: in the absence of a cost, everything should be homogeneous!

If you squint your eyes a bit, you can also recognize a Gibbs free energy minimization problem into this, containing energy, entropy, physical restrictions ($U(\mathbf{a}, \mathbf{b})$) and a temperature ($1/\lambda$). This could be used to describe a system of two types of molecules (for example proteins and ligands) which have a varying degree of cross-affinity for each other.
"""

# ╔═╡ c9cdf082-e986-4a21-8348-cc9faeeeb25a
md"""
### An elegant algorithm for Sinkhorn distances

Even though the entropic regularization can be motivated, to some extent, it appears that we have made the problem harder to solve because we added an extra term. Remarkably, there exists a very simple and efficient algorithm to obtain the optimal distribution matrix $P_\lambda^\star$ and the associated $d_C^\lambda(\mathbf{a}, \mathbf{b})$! This algorithm starts from the observation that the elements of the optimal distribution matrices are of the form

$$(P_\lambda^\star)_{ij} = \alpha_i\beta_j e^{-\lambda C_{ij}}\,,$$

with $\alpha_1,\ldots,\alpha_n$ and $\beta_1,\ldots,\beta_n$ some constants that have to be determined such that the rows, resp. columns, sum to $\mathbf{r}$, resp. $\mathbf{c}$! The optimal distribution matrix can be obtained by the following algorithm.

> **given**: cost matrix $C$, marginals $\mathbf{a}$, $\mathbf{a}$ and $\lambda\ge0$
>
> **initialize**: $P_\lambda = e^{-\lambda C}$
>
> **repeat**
>> 1. **scale the rows** such that the row sums match $\mathbf{a}$
>> 2. **scale the columns** such that the column sums match $\mathbf{b}$
>
> **until** convergence
"""

# ╔═╡ efe892d3-23a3-4545-8c05-7ca9c90a9c81
md"**Assignments**
1. Complete the code below to solve optimal transport using the Sinkhorn algorithm.
2. Solve the dessert problem, once for $\lambda=0.1$ and once for $\lambda=10$."

# ╔═╡ 8214f18f-5d45-455f-b2ce-01085377cdf1
function sinkhorn(C, a, b; λ=1.0, ϵ=1e-8)
    n, m = size(C)
    @assert n == length(a) && m == length(b) throw(DimensionMismatch("a and b do not match"))
    @assert sum(a) ≈ sum(b) "a and b don't have equal sums"
    u, v = copy(a), copy(b)
    M = exp.(-λ .* (C .- maximum(C)))  # substract max for stability
    # normalize this matrix
    while maximum(abs.(a .- Diagonal(u) * (M * v))) > ϵ
        u .= a ./ (M * v)
        v .= b ./ (M' * u)
      end
    return Diagonal(u) * M * Diagonal(v)
  end

# ╔═╡ 425ce568-f3a2-4e73-a6f9-e0373f71436e
# solve the dessert problem for λ=0.1
Plow = sinkhorn(Cdessert, a, b, λ=0.1)

# ╔═╡ f320803a-0439-4881-9570-3b4f6621d044
# solve the dessert problem for λ=10
Phigh = sinkhorn(Cdessert, a, b, λ=10)

# ╔═╡ 0295836f-0672-4903-a3f1-9004a187481b


# ╔═╡ 8d015b52-96a2-4e7a-9827-76a14fe6b58b
md"""

Using this algorithm, we can compute the optimal distribution of desserts, shown below.

![The solution of the dessert problem with $\lambda=10$, an excellent approximation of the unregularized problem.](https://github.com/MichielStock/STMO/blob/master/chapters/06.OptimalTransport/Figures/desserts_high_lamda.png?raw=true)

Here, everybody only has desserts they like. Note that for example, Jan gets three pieces of carrot cake (the only thing he can eat) while Tim receives the remaining portion (he is the only person with some fondness of this dessert). If we decrease the regularization parameter $\lambda$, we encourage a more homogeneous distribution, though some people will have to try some sweets which are not their favorites...

![The solution with a slightly lower $\lambda$. Clearly, a different optimal distribution is obtained.](https://github.com/MichielStock/STMO/raw/master/chapters/06.OptimalTransport/Figures/desserts_low_lamda.png)

The optimal transport problem, with or without entropic regularization has a beautiful geometric interpretation, shown below.

![A geometric view of the optimal transport problem.](https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/06.OptimalTransport/Figures/optimal_transport_geometric.png)

The cost matrix determines a direction in which distributions are better or worse. The set $U(\mathbf{r}, \mathbf{c})$ contains all feasible distributions. In the unregularized case, the optimum $P^\star$ is usually found in one of the corners of such a set. When adding the entropic regularizer, we restrict ourselves to distributions with a minimum of entropy, lying within the smooth red curve. Because we don't have to deal with the sharp corners of $U(\mathbf{r}, \mathbf{c})$ anymore, it is easier to find the optimum. As special cases, when $\lambda\rightarrow \infty$, then $P^\star_\lambda$ will become closers to $P^\star$ (until the algorithm runs into numerical difficulties). For $\lambda\rightarrow 0$ on the other hand, only the entropic term is taken into account and $P_\lambda^\star=\mathbf{a}\mathbf{b}^\intercal$. This is bivariate distribution where the rows and columns are independent.
"""

# ╔═╡ 531be0de-e969-42db-a2fa-f9d125797296
md"""
## Application: color transfer

In this exercise, we will apply optimal transportation to transfer the color scheme of one image to the other. We will read two images and extract the colors of the pixels (RGB encoded). Using optimal transport, a softmatching can be obtained between the pixels of te repective images. Then, every pixel of the first image can be recomputed using a weighted sum of the pixels of the second image (or vice versa).

First, we will load two images. Feel free to use your own!
"""

# ╔═╡ 653a441a-bf0f-4606-a49d-da360d8922dc
download("https://github.com/MichielStock/STMO/blob/master/chapters/06.OptimalTransport/Figures/butterfly3.jpg?raw=true", "image1.jpg")

# ╔═╡ f5d8bd7d-c335-4a10-9cdf-c8f4e953a4ce
download("https://github.com/MichielStock/STMO/blob/master/chapters/06.OptimalTransport/Figures/butterfly2.jpg?raw=true", "image2.jpg")

# ╔═╡ c365eb3f-a7a7-485c-b696-02dd7fccbde3
image1 = load("image1.jpg")  # change to path local directory

# ╔═╡ 648ead24-cd9b-4fe5-97b4-f7a458c9f374
image2 = load("image2.jpg")  # change to path local directory

# ╔═╡ d160a399-7908-48af-9384-c245344e3bdd
md"These images might be large, so let's subsample them!"

# ╔═╡ 8c92dbde-e6a3-42aa-a6a6-b2e615fc9f29
subsample(image, every=8) = image[1:every:size(image,1), 1:every:size(image,2)]

# ╔═╡ 3fb7114f-c765-4153-be6c-66af2d629338
image1ss = subsample(image1)

# ╔═╡ 80e00b5d-3206-4d26-81ed-1228122a60ac
image2ss = subsample(image2)

# ╔═╡ 02b3e106-64e9-4fb4-b4b5-9e1bbe7b8a6d
md"Take a look at the distribution of colors!"

# ╔═╡ 1161be1d-9fcf-45e4-9127-c8560adcca5a
colors1 = vec(image1ss)

# ╔═╡ cc587c5b-bdf1-426a-8733-f4a2ce33637e
colors2 = vec(image2ss)

# ╔═╡ ad3eebaa-4fa2-438a-a088-16334dba3a89
n_colors1 = length(colors1)

# ╔═╡ 11deb453-bd6e-43b1-b1e1-61606644eef8
n_colors2 = length(colors2)

# ╔═╡ e52d67d0-b161-403a-85c6-1bb59354f2ee
md"The `Colors` package contains a function `colordiff` that quantifies the perceptive difference between two colors. It is an ideal cost function!"

# ╔═╡ dfe40afd-5153-4315-8511-cba00806d7e5
md"Use this to compute the pairwise distances between all the colors!"

# ╔═╡ 7733a347-47bc-4a98-ac54-7139a945de0c
Ccol = [colordiff(c1, c2) for c1 in colors1, c2 in colors2]

# ╔═╡ 0d719be9-aa39-4d12-9530-d50feb179753
md"Now, each pixel of one image is soft-matched to pixels of the other picture. Recolor each pixel based on the weighted average of its assigned pixels to do the color transfer."

# ╔═╡ 84470c18-41f7-4ed8-b6a8-25eefbc97e41
a_col = ones(n_colors1) / n_colors1

# ╔═╡ 6be95f94-e71f-4447-bc7a-cf1bfcda49a4
b_col = ones(n_colors2) / n_colors2

# ╔═╡ 49fd9af6-f5a5-4f82-a39d-39f741430df6
Pcolors = Solution.sinkhorn(Ccol,
			a_col,
			b_col, λ=13, ϵ=1e-13)

# ╔═╡ 96792a45-2a1f-48de-a514-e5dfd97aabf7
md"Then, we can compute the transport between the two collections of pixels. Since every pixel is equally important, we give an uniform weight for each pixel, e.g., $\mathbf{a}=\mathbf{1}_n/m$ and $\mathbf{b}=\mathbf{1}_m/m$"

# ╔═╡ fa81b789-a1ec-42cc-b73f-2fb1e6d343c3
"""
Maps one distribution to the other
"""
mapdistr(X, P) = Diagonal(sum(P, dims=2)[:].^-1) * P * X

# ╔═╡ e07b9f46-3cbe-490e-8e42-60470db2d4b4
image1_transf = reshape(mapdistr(colors2, Pcolors), size(image1ss))

# ╔═╡ 64dcfee5-941c-417f-9698-18f57e4887f6
image2_transf = reshape(mapdistr(colors1, Pcolors'), size(image2ss))

# ╔═╡ be1024ce-78fb-4436-9771-d1d952bfa69b
md"## References

- Gabriel Peyré and Marco Cuturi, *Computational Optimal Transport*, ArXiv:1803.00567, 2018.
"

# ╔═╡ 15da4690-13fd-4b05-ac12-b5ccc2898bd0
begin
	myblue = "#304da5"
	mygreen = "#2a9d8f"
	myyellow = "#e9c46a"
	myorange = "#f4a261"
	myred = "#e76f51"
	myblack = "#50514F"

	mycolors = [myblue, myred, mygreen, myorange, myyellow]
end;

# ╔═╡ 0b50b550-8c2b-43d7-849b-73ee8f60ff52
begin
	ha = bar(a, color=mygreen, title="staff");
	xticks!(1:8, staff)
	hb = bar(b, color=myorange, title="desserts");
	xticks!(1:5, desserts)
	plot(ha, hb)
end

# ╔═╡ 49b390b5-2192-43d9-ac71-337e2d19799b
myr, myg, myb = parse(Colorant, myred), parse(Colorant, mygreen), parse(Colorant, myblue)

# ╔═╡ 49193509-14ea-4f0f-8fc1-14e7a924d2d7
colordiff(myr, myg)  # difference between red and green

# ╔═╡ 627d6a6f-1c20-4242-8fc8-0a6a2ed91549
colordiff(myb, myg)  # difference between blue and green

# ╔═╡ 086947e8-1baf-43c6-9c64-f10187b0a9c4
X1 = [-3.334062779243765 -0.4501346178106358;
-17.089568765347632 13.122166467239909;
-14.347484681558905 -6.733978925249632;
4.023051337264342 -9.962304692857852;
-8.463617750898711 -7.272060919414795;
4.411632356371395 18.33766410251966;
0.33231514398722767 -8.943403560908134;
4.312466564646311 -4.769277598451736;
-0.1530439721692118 9.400009756002328;
-5.9829675943508995 0.07220200778660099];

# ╔═╡ ba9e51ae-f9d5-4ffe-af79-7bff04ae41b1
X2 = [-15.776290315139038 -5.26704453200824;
5.035133784625716 6.388525577583841;
-1.302878737641465 -14.087978411169559;
-17.05014605333986 10.189492174622178;
-10.102710138685817 1.6112603598816224;
-17.880416477734084 -5.15108311036524;
-18.49092111472347 -5.193032093476489;
1.2972905950753137 34.72586483583929;
1.3580992685186684 7.334727533023157;
-7.482909025521323 6.326317440680775];

# ╔═╡ 1104b6a8-e781-4f58-ba2d-798ebbc11422
begin
	p_cells = scatter(X1[:,1], X1[:,2], color=myorange,
		label="location cells at t1", legend=:topleft)
	xlabel!("\$x\$")
	ylabel!("\$y\$")
	scatter!(X2[:,1], X2[:,2], color=mygreen, label="location cells at t2")
	
	p_cells
end

# ╔═╡ 31627338-8e0b-4568-a85b-3d7414c87769
begin
	"""
	Compute Euclidean distance between two vectors.
	"""
	dist(x::AbstractVector, y::AbstractVector) = sqrt(sum((x .- y).^2))
	"""
	Compute Euclidean distance matrix between two matrices.
	"""
	dist(X::AbstractMatrix, Y::AbstractMatrix) = [dist(X[i,:], Y[j,:]) for i in 1:size(X,1), j in 1:size(Y,1)]
	"""
	Compute Euclidean distance matrix.
	"""
	dist(X::AbstractMatrix) = dist(X::AbstractMatrix, X::AbstractMatrix)
end

# ╔═╡ 1468ec48-bedb-4977-8b88-794d89e5ae8f
C_cells = dist(X1, X2).^2  # squared distance

# ╔═╡ 88569ef5-bc60-4a77-aa72-fc9bdf66b3a4
best_perm, best_score = monge_brute_force(C_cells)

# ╔═╡ f254b8a9-45a8-4f32-8929-670dd5140ce8
let
	p_cells = scatter(X1[:,1], X1[:,2], color=myorange,
		label="location cells at t1", legend=:topleft)
	xlabel!("\$x\$")
	ylabel!("\$y\$")
	scatter!(X2[:,1], X2[:,2], color=mygreen, label="location cells at t2")
	for (i, σᵢ) in enumerate(best_perm)
		plot!(p_cells, [X1[i,1], X2[σᵢ,1]], [X1[i,2], X2[σᵢ,2]], color=myred, label="")
	end
	
	p_cells
end

# ╔═╡ 92f53b57-8de4-4711-a049-f266f0d621da
monge_brute_force_oneline(C_cells)

# ╔═╡ 3070288a-8060-4da0-8007-41b69c606521
colorscatter(colors; kwargs...) = scatter(red.(colors), green.(colors), blue.(colors),
                        xlabel="red", ylabel="green", zlabel="blue", color=colors, label="")

# ╔═╡ ef606085-c423-4287-b4db-62ce3ddd0ed7
plot(
  colorscatter(colors1, title="figure 1"),
  colorscatter(colors2, title="figure 2"))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Colors = "~0.12.8"
Combinatorics = "~1.0.2"
ImageIO = "~0.5.8"
Images = "~0.24.1"
Plots = "~1.22.1"
PlutoUI = "~0.7.10"
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

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

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

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "45efb332df2e86f2cb2e992239b6267d97c9e0b6"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6d1c23e740a586955645500bbec662476204a52c"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.1"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

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

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

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

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "70a0cfd9b1c86b0209e38fbfe6d8231fd606eeaf"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.1"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

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
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

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

[[IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageAxes]]
deps = ["AxisArrays", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "794ad1d922c432082bc1aaa9fa8ffbd1fe74e621"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.9"

[[ImageContrastAdjustment]]
deps = ["ColorVectorSpace", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "2e6084db6cccab11fe0bc3e4130bd3d117092ed9"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.7"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageDistances]]
deps = ["ColorVectorSpace", "Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6378c34a3c3a216235210d19b9f495ecfff2f85f"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.13"

[[ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[ImageMetadata]]
deps = ["AxisArrays", "ColorVectorSpace", "ImageAxes", "ImageCore", "IndirectArrays"]
git-tree-sha1 = "ae76038347dc4edcdb06b541595268fca65b6a42"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.5"

[[ImageMorphology]]
deps = ["ColorVectorSpace", "ImageCore", "LinearAlgebra", "TiledIteration"]
git-tree-sha1 = "68e7cbcd7dfaa3c2f74b0a8ab3066f5de8f2b71d"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.2.11"

[[ImageQualityIndexes]]
deps = ["ColorVectorSpace", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1198f85fa2481a3bb94bf937495ba1916f12b533"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.2.2"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[Images]]
deps = ["AxisArrays", "Base64", "ColorVectorSpace", "FileIO", "Graphics", "ImageAxes", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageShow", "ImageTransformations", "IndirectArrays", "OffsetArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "8b714d5e11c91a0d945717430ec20f9251af4bd2"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.24.1"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

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

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

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

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

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

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e14c485f6beee0c7a8dcf6128bf70b85f1fe201e"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.9"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

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

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

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

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

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

[[Rotations]]
deps = ["LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "2ed8d8a16d703f900168822d83699b8c3c1a5cd8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.2"

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

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

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

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "632a8d4dbbad6627a4d2d21b1c6ebcaeebb1e1ed"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.2"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "52c5f816857bfb3291c7d25420b1f4aca0a74d18"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

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
# ╟─6f9362d0-1bb6-11ec-2b8b-491ef1f091e5
# ╠═3ea4f8be-d0a7-455d-b58e-3e81ea61f9c1
# ╟─ccb9bf05-1830-4c56-bf28-312c5d324484
# ╟─65a73b72-967d-4a62-af17-f71b286f3303
# ╠═ed3f476c-2205-43a7-a4cf-5941a0d0afd0
# ╠═0e579d2e-9b78-4a56-a30d-883cec0ad5d6
# ╠═ce639d43-7481-43f4-8724-1da9c18d716a
# ╠═c32441ee-c62d-4ee7-8e1b-ff7de1072d43
# ╟─0b50b550-8c2b-43d7-849b-73ee8f60ff52
# ╟─245acb05-d58b-4ca6-9cf4-e59ff50e86c9
# ╠═114c8c91-34f3-4256-ab0f-c1fbd64e4827
# ╠═01e3b967-e441-4041-8bd8-5cfc0e0ebaa4
# ╟─7dbf484a-f3fb-4510-b500-57f4b13c0f59
# ╟─b3440682-8443-4052-bbf2-8f593c4e54c3
# ╟─984effce-9a76-4bbb-9105-06887dfa629e
# ╟─9567f11f-486a-4960-8b69-42d2e5ee413f
# ╟─1104b6a8-e781-4f58-ba2d-798ebbc11422
# ╠═b85b69fb-bfdb-41bd-a6a1-1aebe3077e08
# ╠═cc62b7b1-fe34-4a19-846d-ca6c7fb69ff9
# ╠═1468ec48-bedb-4977-8b88-794d89e5ae8f
# ╠═88569ef5-bc60-4a77-aa72-fc9bdf66b3a4
# ╠═f254b8a9-45a8-4f32-8929-670dd5140ce8
# ╟─4bf0f9d1-806c-40e2-8ef4-325be3a5b2ca
# ╠═6285291e-0520-40f1-b39e-a98991995e47
# ╠═92f53b57-8de4-4711-a049-f266f0d621da
# ╟─da2cc178-64c7-4b30-909a-24fb402a489b
# ╟─bdd25212-576e-4b81-8178-1212ac759cbc
# ╟─c9cdf082-e986-4a21-8348-cc9faeeeb25a
# ╟─efe892d3-23a3-4545-8c05-7ca9c90a9c81
# ╠═8214f18f-5d45-455f-b2ce-01085377cdf1
# ╠═425ce568-f3a2-4e73-a6f9-e0373f71436e
# ╠═f320803a-0439-4881-9570-3b4f6621d044
# ╠═0295836f-0672-4903-a3f1-9004a187481b
# ╟─8d015b52-96a2-4e7a-9827-76a14fe6b58b
# ╟─531be0de-e969-42db-a2fa-f9d125797296
# ╠═653a441a-bf0f-4606-a49d-da360d8922dc
# ╠═f5d8bd7d-c335-4a10-9cdf-c8f4e953a4ce
# ╠═c365eb3f-a7a7-485c-b696-02dd7fccbde3
# ╠═648ead24-cd9b-4fe5-97b4-f7a458c9f374
# ╟─d160a399-7908-48af-9384-c245344e3bdd
# ╠═8c92dbde-e6a3-42aa-a6a6-b2e615fc9f29
# ╠═3fb7114f-c765-4153-be6c-66af2d629338
# ╠═80e00b5d-3206-4d26-81ed-1228122a60ac
# ╟─02b3e106-64e9-4fb4-b4b5-9e1bbe7b8a6d
# ╠═1161be1d-9fcf-45e4-9127-c8560adcca5a
# ╠═cc587c5b-bdf1-426a-8733-f4a2ce33637e
# ╠═ad3eebaa-4fa2-438a-a088-16334dba3a89
# ╠═11deb453-bd6e-43b1-b1e1-61606644eef8
# ╠═ef606085-c423-4287-b4db-62ce3ddd0ed7
# ╟─e52d67d0-b161-403a-85c6-1bb59354f2ee
# ╠═49b390b5-2192-43d9-ac71-337e2d19799b
# ╠═49193509-14ea-4f0f-8fc1-14e7a924d2d7
# ╠═627d6a6f-1c20-4242-8fc8-0a6a2ed91549
# ╟─dfe40afd-5153-4315-8511-cba00806d7e5
# ╠═7733a347-47bc-4a98-ac54-7139a945de0c
# ╟─0d719be9-aa39-4d12-9530-d50feb179753
# ╠═84470c18-41f7-4ed8-b6a8-25eefbc97e41
# ╠═6be95f94-e71f-4447-bc7a-cf1bfcda49a4
# ╠═49fd9af6-f5a5-4f82-a39d-39f741430df6
# ╟─96792a45-2a1f-48de-a514-e5dfd97aabf7
# ╠═fa81b789-a1ec-42cc-b73f-2fb1e6d343c3
# ╠═e07b9f46-3cbe-490e-8e42-60470db2d4b4
# ╠═64dcfee5-941c-417f-9698-18f57e4887f6
# ╟─be1024ce-78fb-4436-9771-d1d952bfa69b
# ╟─15da4690-13fd-4b05-ac12-b5ccc2898bd0
# ╟─086947e8-1baf-43c6-9c64-f10187b0a9c4
# ╟─ba9e51ae-f9d5-4ffe-af79-7bff04ae41b1
# ╟─31627338-8e0b-4568-a85b-3d7414c87769
# ╟─3070288a-8060-4da0-8007-41b69c606521
# ╟─8de53f65-0d4a-4af0-a890-609ddcc92b36
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
