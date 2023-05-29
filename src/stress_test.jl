include("PerceptronNetwork.jl")
include("DataIO.jl")
include("NetworkIO.jl")

using .PerceptronNetwork
using .DataIO
using .NetworkIO

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
]
training_data = get_feature_vector_tuples(
    "data/input/product_training.csv",
    feature_cols,
    "awesomeness",
    :
)

println("extracted training data\ntraining model...\n")

b, W = train_network(
    training_data,
    2,
    500,
    10,
    50,
    [10, 50, 50, 2],
    0.01
)

println("training complete\n")

write_network("data/output/product_nn.ser", b, W)

println("serialized network\npredicting output...\n")

predictions = predict(
  [sample[1] for sample in training_data],
  b,
  W
)

println("predicting complete\n")

prediction_df = read_id_column("data/input/product_training.csv", "asin")
generate_prediction_df!(predictions, prediction_df, "awesomeness")
save_predictions("data/output/product_nn_output.json", prediction_df)

println("saved predictions")
