include("PerceptronNetwork.jl")
include("CSVReader.jl")

using .PerceptronNetwork
using .CSVReader

features = get_feature_vector("data/test_data.csv", ["A", "B"])
println(features)

truths = get_feature_vector("data/test_data.csv", ["C"])
println(truths)

A, Z, W, B = train_network(features, truths, [(2, 2), (2, 2), (1, 2)], 0.01, 1e6)