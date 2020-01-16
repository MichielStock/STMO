module OptimalTransport

# MONGE PROBLEM
# -------------

using Combinatorics: permutations

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

using LinearAlgebra

function sinkhorn(C::Matrix, a::Vector, b::Vector; λ=1.0, ϵ=1e-8)
    n, m = size(C)
    @assert n == length(a) && m == length(b) throw(DimensionMismatch("a and b do not match"))
    @assert sum(a) ≈ sum(b) "a and b don't have equal sums"
    u, v = copy(a), copy(b)
    M = exp.(-λ * C)
    # normalize this matrix
    while maximum(abs.(a .- Diagonal(u) * (M * v))) > ϵ
        u .= a ./ (M * v)
        v .= b ./ (M' * u)
      end
    return Diagonal(u) * M * Diagonal(v)
  end

export sinkhorn

end
