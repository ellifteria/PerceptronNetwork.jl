module CSVReader

using DataFrames, CSV

export get_feature_vector

function get_feature_vector(csv_path::String, columns::Vector)::Vector{Vector}
    df = DataFrame(CSV.File(csv_path))
    df_subset = df[:, columns]
    M = transpose(Matrix(df_subset))
    feature_vector = [M[:,i] for i in 1:size(M,2)]
    return feature_vector
end

end