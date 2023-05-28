include("PerceptronNetwork.jl")
include("CSVReader.jl")

using .PerceptronNetwork

training_data = [
    ([2,11],2),
    ([9,12],2),
    ([6,10],2),
    ([5,8],2),
    ([7,8],2),
    ([3,9],1),
    ([2,7],1),
    ([1,2],1),
    ([4,4],1),
    ([6,2],1),
    ([10,6],1),
    ([11,3],2)
]

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

# predict(
#     [sample[1] for sample in training_data],
#     b,
#     W
# )

