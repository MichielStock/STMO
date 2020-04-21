@testset "TSP" begin

using STMO.TSP

X = [1 1; 1 3; 2 2; 2 1]

tsp = TravelingSalesmanProblem(X)

@test tsp isa TravelingSalesmanProblem{Int,Float64}

tour = [4, 1, 2, 3]

@test Set(cities(tsp)) == Set(tour)

@test isvalid(tsp, tour)
@test !isvalid(tsp, [1, 2, 3])
@test !isvalid(tsp, [1, 2, 3, 4, 5])
@test !isvalid(tsp, [3, 1, 2, 3])

@test dist(tsp, 2, 3) ≈ sqrt(2)
@test computecost(tsp, tour) ≈ 2 + sqrt(2) + 1 + 1


end
