include("PerceptronNetwork.jl")
include("DataIO.jl")
include("NetworkIO.jl")

using .PerceptronNetwork
using .DataIO
using .NetworkIO

MODELPATH = "data/output/product_nn.ser"

DATAPATH = "data/input/product_training.csv"

OUTPUTPATH = "data/output/product_nn_output.json"

FEATURECOLS = [
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

DATAID = "asin"
PREDICTIONID = "awesomeness"

println("reading serialized model...\n")

b, W = read_network(MODELPATH)

println("read serialized network\n\nextracting input data...")

input_data = get_feature_vector_tuples(
    DATAPATH,
    FEATURECOLS,
    FEATURECOLS[1],
    :
)

println("extracted data\n\npredicting output...\n")

predictions = predict(
  [sample[1] for sample in input_data],
  b,
  W
)

println("predicting complete\n")

prediction_df = read_id_column(DATAPATH, DATAID)
generate_prediction_df!(predictions, prediction_df, PREDICTIONID)
save_predictions(OUTPUTPATH, prediction_df)

println("saved predictions")
