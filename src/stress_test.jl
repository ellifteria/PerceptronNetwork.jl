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

