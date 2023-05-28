module CSVReader

using DataFrames, CSV

export get_feature_vector_tuples

function get_feature_vector_tuples(
    csv_path::String,
    feature_columns::Vector,
    truth_column::String,
    rows=:
)
    df = DataFrame(CSV.File(csv_path))
    return [
        (
            Vector(
                df[row, feature_columns]
            ),
            df[row, truth_column]
        )
        for row in 1:size(df[rows, :])[1]
    ]
end

end
