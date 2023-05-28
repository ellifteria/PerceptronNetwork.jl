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
    "avg_neg_votes"
    # "unixReviewTime"
]
training_data = get_feature_vector_tuples(
    "data/product_training.csv",
    feature_cols,
    "awesomeness",
    :
   )
println("extracted training data")

# println(training_data)
b, W = train_network(
    training_data,
    2,
    100,
    1,
    1,
    [10, 50, 50, 2],
    0.01
)

predict(
    [sample[1] for sample in training_data],
    b,
    W
)
