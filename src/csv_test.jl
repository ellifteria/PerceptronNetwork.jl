include("PerceptronNetwork.jl")
include("CSVReader.jl")

using .PerceptronNetwork
using .CSVReader

feature_cols = [
    "x1",
    "x2"
]

training_data = get_feature_vector_tuples(
    "data/test_data.csv",
    feature_cols,
    "y",
    :
   )
println("extracted training data")


b, W = train_network(
    training_data,
    2,
    500,
    100,
    4,
    [2, 50, 50, 2],
    0.1
)

println("training complete")

