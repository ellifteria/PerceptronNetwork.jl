module PerceptronNetwork

using Random

export train_network, feed_forward, compute_accurary,
    predict

function quadratic_cost(z, a, Y)
    return (a .- Y) .* dsigmoid(z)
end

function relu(x)
    return (x .> 0) .* x
end

function drelu(x)
    return (x .> 0)
end

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-1 .* x))
end

function dsigmoid(x)
    return exp.(-1 .* x) ./ ((1 .+ exp.(-1 .* x)) .^ 2)
end

function initialize_weights_biases(sizes)
    biases = [randn(l_size, 1) for
        l_size in sizes[2:end]]
    weights = [randn(l_size1, l_size2) for
        (l_size1, l_size2) in zip(sizes[2:end], sizes[1:end-1])]
    return biases, weights
end

function feed_forward(
    X,
    biases,
    weights,
    activation_function=sigmoid
)

    a = copy(X)
    as = Vector{VecOrMat}([a])
    zs = Vector{VecOrMat}([])

    for (b, w) in zip(biases, weights)
        z = (w * a) + b
        a = activation_function(z)

        push!(zs, z)
        push!(as, a)
    end
    return a, as, zs
end

function propagate_back(
    X,
    Y,
    biases,
    weights,
    activation_function=sigmoid,
    cost_function=quadratic_cost,
    dactivation_function=dsigmoid
)

    num_layers = length(biases)

    nabla_b = [zeros(size(b)) for b in biases]
    nabla_W = [zeros(size(W)) for W in weights]

    _, as, zs = feed_forward(
        X, biases, weights, activation_function
    )

    delta = cost_function(zs[end], as[end], Y)
    nabla_b[end] = delta
    nabla_W[end] = delta * as[end-1]'

    for l = 2:num_layers
        delta = (weights[end+2 - l]' * delta) .*
            dactivation_function(zs[end+1 - l])
        nabla_b[end+1 - l] = delta
        nabla_W[end+1 - l] = delta * as[end - l]'
    end

    return nabla_b, nabla_W
end

function update_network(
    training_set,
    biases,
    weights,
    eta,
    activation_function=sigmoid,
    cost_function=quadratic_cost,
    dactivation_function=dsigmoid
)

    num_samples = length(training_set)

    nabla_b = [zeros(size(b)) for b in biases]
    nabla_W = [zeros(size(W)) for W in weights]

    for (X, Y) in training_set
        dnabla_b, dnabla_W = propagate_back(
            X, Y, biases, weights, 
            activation_function, cost_function,
            dactivation_function)
        nabla_b = [nab_b .+ dnab_b for
            (nab_b, dnab_b) in zip(nabla_b, dnabla_b)]
        nabla_W = [nab_W .+ dnab_W for
            (nab_W, dnab_W) in zip(nabla_W, dnabla_W)]
    end

    biases = [b .- (eta/num_samples) .* nab_b for
        (b, nab_b) in zip(biases, nabla_b)]
    weights = [W .- (eta/num_samples) .* nab_W for
        (W, nab_W) in zip(weights, nabla_W)]
    return biases, weights
end

function generate_predictions(Y_hat)
    return [argmax(vec(y_hat)) for y_hat in Y_hat]
end

function compute_accurary(ground_truths, predictions)
    return sum(ground_truths .== predictions)/length(predictions)
end

function vectorize_output(output, num_classes)
    return 1:num_classes .== output
end

function vectorize_outputs(outputs, num_classes)
    vectorized_output = [vectorize_output(output, num_classes) for output in outputs]
    return vectorized_output
end

function vectorize_training_outputs(training_data, num_classes)
    return [
        (
            row[1],
            vectorize_output(row[2], num_classes)
        )
        for row in training_data
    ]
end

function train_network(
    training_data,
    num_classes,
    epochs,
    print_frequency,
    batch_size,
    network_sizes,
    eta,
    activation_function=sigmoid,
    cost_function=quadratic_cost,
    dactivation_function=dsigmoid,
    needs_vectorizing=true
)
    
    num_samples = length(training_data)

    biases, weights =
        initialize_weights_biases(network_sizes)

    X = [sample[1] for sample in training_data]
    Y = [sample[2] for sample in training_data]

    ground_truths = generate_predictions(
        vectorize_outputs(Y, num_classes)
    )

    if needs_vectorizing
        training_data = vectorize_training_outputs(
            training_data,
            num_classes
        )
    end

    for epoch = 1:epochs
        shuffled = shuffle(training_data)
        batches = [shuffled[
            start:start+batch_size] for
            start = 1:batch_size:num_samples-batch_size]
        for batch in batches
            biases, weights = update_network(
                batch,
                biases,
                weights,
                eta,
                activation_function,
                cost_function,
                dactivation_function
            )
        end
        
        if epoch % print_frequency == 0 ||
            epoch == 1
            println("Epoch $epoch complete")

            network_predictions = predict(
                X,
                biases,
                weights,
                activation_function
            )

            println("Accuracy: $(
                compute_accurary(
                    ground_truths,
                    network_predictions
                )
            )\n")
        end
    end

    return biases, weights
end

function predict(
    X,
    biases,
    weights,
    activation_function=sigmoid
)

    raw_predictions = [
        feed_forward(
            x, biases, weights, activation_function
        )[1] for x in X
    ]

    predictions = generate_predictions(
        raw_predictions
    )

    return predictions
end

end
