

module CancerData

export getcancerdata

using HTTP, CSV


"""
    getcancerdata()

Gets the cancer dataset for binary classification. Returns `features` and
`binary_response`. The feature matrix is standardized, the first column is a
vector of ones as the intercept. The binary responses have true denote Malignant
and false Benign. Internet connection is needed for this function.
"""
function getcancerdata()
    io = HTTP.get("https://raw.githubusercontent.com/MichielStock/STMO/master/chapters/03.Unconstrained/Data/BreastCancer.csv")
    cancer_data = CSV.read(io.body)
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


end #Words
