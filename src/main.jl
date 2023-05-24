include("PerceptronNetwork.jl")
include("CSVReader.jl")

using .PerceptronNetwork
using .CSVReader

feature_cols = [
    "x1",
    "x2"
]
features = get_feature_vector(
    "data/test_data.csv",
    feature_cols
    )
println("extracted training features: size=$(length(features)) x $(length(features[1]))")

truth_cols = [
    "y"
]
truths = get_feature_vector(
    "data/test_data.csv",
    truth_cols
    )
println("extracted training truths: size=$(length(truths)) x $(length(truths[1]))")

A, Z, W, B = train_network(
    features,
    truths,
    [
        (5, length(feature_cols)),
        (5, 5),
        (5, 5),
        (5, 5),
        (length(truth_cols), 5)
    ],
    1e-7,
    1e4,
    1e2
    )

println("final error = $(total_loss(truths, A))")
println("final accuracy = $(calculate_accuracy(truths, get_predictions(A)))")
println("predictions:\n$(get_predictions(A))")

