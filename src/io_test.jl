include("PerceptronNetwork.jl")
include("DataIO.jl")
include("NetworkIO.jl")

using .PerceptronNetwork
using .DataIO
using .NetworkIO

feature_cols = [
    "x1",
    "x2"
]

training_data = get_feature_vector_tuples(
    "data/input/test_data.csv",
    feature_cols,
    "y",
    :
)
   
println("extracted training data\ntraining model...\n")


b, W = train_network(
    training_data,
    2,
    500,
    100,
    4,
    [2, 50, 50, 2],
    0.1
)

println("training complete\n")

write_network("data/output/io_test.ser", b, W)

println("serialized network\n")

serialized_b, serialized_W = read_network("data/output/io_test.ser")

println("read serialized network\npredicting output...\n")

predictions = predict(
  [sample[1] for sample in training_data],
  serialized_b,
  serialized_W
)

println("predictions:")
println(predictions)

prediction_df = read_id_column("data/input/test_data.csv")
generate_prediction_df!(predictions, prediction_df)
save_predictions("data/output/io_test.json", prediction_df)

println("\nsaved predictions")
