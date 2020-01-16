#=
Created on Thursday 16 Jan 2020
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Solution for the color transfer.
=#

using STMO
using STMO.OptimalTransport
using Plots, LaTeXStrings

using Images, Colors

# function to subsample image
subsample(image, every=8) = image[1:every:size(image,1), 1:every:size(image,2)]

# load images
image1 = load("chapters/06.OptimalTransport/Figures/butterfly3.jpg")
image2 = load("chapters/06.OptimalTransport/Figures/butterfly2.jpg")

# subsample
image1 = subsample(image1, 8)
image2 = subsample(image2, 5)

# get colors, can also using sampling
colors1 = vec(image1)
colors2 = vec(image2)

# distance matrix
C = [colordiff(c1, c2) for c1 in colors1, c2 in colors2]

n, m = size(C)

# optimal transport
P = sinkhorn(C, ones(n)/n, ones(m)/m, λ=10)

mapdistr(X, P) = Diagonal(sum(P, dims=2)[:].^-1) * P * X

# map colors
image1transf = reshape(mapdistr(colors2, P), size(image1)...)
image2transf = reshape(mapdistr(colors1, P'), size(image2)...)

plot(
plot(image1, title="image 1", aspect_ratio=:equal),
plot(image2, title="image 2", aspect_ratio=:equal),
plot(image1transf, title="image 1 transfered", aspect_ratio=:equal),
plot(image2transf, title="image 2 transfered", aspect_ratio=:equal)
)

savefig("figures/colortransfer")
