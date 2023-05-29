module DataIO

using DataFrames, CSV, JSON

export get_feature_vector_tuples, save_predictions, generate_prediction_df!, read_id_column

function get_feature_vector_tuples(
    file_path::String,
    feature_columns::Vector,
    truth_column::String,
    rows=:,
    increment_classes_by_1::Bool=true
)
    df = DataFrame(CSV.File(file_path))
    return [
        (
            Vector(
                df[row, feature_columns]
            ),
            df[row, truth_column] .+ (1 * increment_classes_by_1)
        )
        for row in 1:size(df[rows, :])[1]
    ]
end

function generate_prediction_df!(
  predictions::Vector,
  id_dataframe::DataFrame,
  prediction_col_name::String="Prediction",
  decrement_predictions_by_1::Bool=true
)
  id_dataframe[!, prediction_col_name] = predictions .- (1 * decrement_predictions_by_1)
end

function read_id_column(file_path::String, id_column_name::String="id")
  df = DataFrame(CSV.File(file_path))
  return df[:, [id_column_name]]
end

function save_predictions(file_path::String, predictions::DataFrame)
  open(file_path, "w") do f
    write(f, JSON.json(predictions))
  end
end

end
