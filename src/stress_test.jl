include("PerceptronNetwork.jl")
include("CSVReader.jl")

using .PerceptronNetwork
using .CSVReader

feature_cols = [
    "summary_compound",
    "reviewText_compound",
    "positive_pct",
    "negative_pct",
    "pct_pos_w_img",
    "pct_neg_w_img",
    "pct_pos_ver",
    "pct_neg_ver",
    "avg_pos_votes",
    "avg_neg_votes",
    "unixReviewTime"
]
features = get_feature_vector(
    "data/product_training.csv",
    feature_cols
    )
println("extracted training features: size=$(length(features)) x $(length(features[1]))")

truth_cols = [
    "awesomeness"
]
truths = get_feature_vector(
    "data/product_training.csv",
    truth_cols
    )
println("extracted training truths: size=$(length(truths)) x $(length(truths[1]))")

A, Z, W, B = train_network(
    features,
    truths,
    [
        (100, length(feature_cols)),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (100, 100),
        (length(truth_cols), 100)
    ],
    1e-93,
    1e3,
    1e0
    )

println("final error = $(total_loss(truths, A))")
println("final accuracy = $(calculate_accuracy(truths, get_predictions(A)))")

