#=
Created on Monday 13 January 2020
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Test functions for optimization
=#

module TestFuns

using LinearAlgebra

function ackley(x, a=20, b=0.2, c=2π)
    d = length(x)
    return -a * exp(-b*sqrt(sum(x.^2)/d)) -
        exp(sum(cos.(c .* x))/d)
end

# Branin function
function branin((x1,x2); a=1, b=5.1/(4pi^2), c=5/pi, r=6, s=10, t=1/8pi)
    return a * (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s
end

booth((x1, x2)) = (x1+2x2-7)^2 + (2x1+x2-5)^2

# Rosenbrock
rosenbrock((x1, x2); a=1, b=5) = (a-x1)^2 + b*(x2-x1^2)^2

rastrigine(x; A=10) = length(x) * A + sum(x.^2 .+ A * cos.(2pi*x))

function flower(x; a=1, b=1, c=4)
    return a * norm(x) + b * sin(c*atan(x[2], x[1]))
end

export ackley, branin, rosenbrock, rastrigine, flower, booth

# functions for convex optimization

fquadr((x1, x2); γ=10.0) = 0.5(x1^2 + γ * x2^2);
grad_fquadr((x1, x2); γ=10.0) = [x1, γ * x2];
hess_fquadr((x1, x2); γ=10.0) = [1 0; 0 γ];

export fquadr, grad_fquadr, hess_fquadr

import Zygote  # NOT doing this by hand!

fnonquadr((x1, x2)) = log(exp(x1+3x2-0.1) + exp(x1-3x2-0.1)+exp(-x1-0.1));
grad_fnonquadr(x) = Zygote.gradient(fnonquadr, x)[1];
hess_fnonquadr(x) = Zygote.hessian(fnonquadr, x);

export fnonquadr, grad_fnonquadr, hess_fnonquadr

end  # module TestFuns
