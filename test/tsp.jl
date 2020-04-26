@testset "TSP" begin

using STMO.TSP

X = [1 1; 1 3; 2 2; 2 1]

tsp = TravelingSalesmanProblem(X)

@test tsp isa TravelingSalesmanProblem{Int}

tour = [4, 1, 2, 3]

n = length(tour)

@test n == 4

@test Set(cities(tsp)) == Set(tour)

@test isvalid(tsp, tour)
@test !isvalid(tsp, [1, 2, 3])
@test !isvalid(tsp, [1, 2, 3, 4, 5])
@test !isvalid(tsp, [3, 1, 2, 3])

@test dist(tsp, 2, 3) ≈ sqrt(2)
@test computecost(tsp, tour) ≈ 2 + sqrt(2) + 1 + 1

@testset "swapping" begin
    cost = computecost(tsp, tour)

    for i in 1:n
        for j in 1:n
            Δc = deltaswapcost(tsp, tour, i, j)
            swap!(tour, i, j)
            @test computecost(tsp, tour) ≈ cost + Δc
            swap!(tour, i, j)
        end
    end
end

@testset "flipping" begin
    cost = computecost(tsp, tour)

    for i in 1:n
        for j in 1:n
            Δc = deltaflipcost(tsp, tour, i, j)
            flip!(tour, i, j)
            @test computecost(tsp, tour) ≈ cost + Δc
            flip!(tour, i, j)
        end
    end
end

end
