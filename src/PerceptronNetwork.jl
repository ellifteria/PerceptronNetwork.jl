module PerceptronNetwork

using Statistics

export relu, drelu, feed_forward, propogate_back, train_network, loss

function relu(x)
  return (x .> 0) .* x
end

function drelu(x)
  return (x .> 0)
end

function feed_forward(inputs, weights, biases)
  layers = length(weights)
  z = Vector{Vector}(undef, layers)
  a = Vector{Vector}(undef, layers + 1)
  a[1] = inputs
  for l = 1:layers
    z[l] = vec(weights[l]*a[l] .+ biases[l])
    a[l+1] = vec(relu(z[l]))
  end
    return a, z
end

function propogate_back(y, a, z, weights)
  layers = length(weights)
  # println(a[end])
  # println(y)
  nablaC = a[end] .- y
  delta = Vector{VecOrMat}(undef, layers)
  delta[layers] = nablaC .* drelu(z[layers])
  for l = layers-1:-1:1
    delta[l] = (transpose(weights[l+1]) * delta[l+1]) .* drelu(z[l])
  end
  return delta
end

function update_weights_and_biases(a, weights, biases, delta, eta)
  layers = length(weights)
  new_biases = copy(biases)
  new_weights = copy(weights)
  for l = layers:-1:1
    new_biases[l] = vec(biases[l] .- delta[l] .* eta)
    new_weights[l] = weights[l] .- (delta[l] * transpose(a[l])) .* eta
  end
  return new_weights, new_biases
end

function loss(y, y_hat)
  return (1/2) .* (y_hat .- y) .^ 2
end

function sum_square_error(Y, Y_hat)
  println(Y)
  println(Y_hat)
  sum_errors = 0
  for i = 1:length(Y)
    sum_errors .+= loss(Y[i], Y_hat[i])
  end
end

function train_network(inputs, y, layer_shapes, eta, iterations, print_frequency = 100)
  if !(inputs[1] isa VecOrMat)
    inputs = [inputs]
  end
  if !(y[1] isa VecOrMat)
    y = [y]
  end
  layers = length(layer_shapes)
  weights = Vector{Matrix}(undef, layers)
  biases = Vector{Vector}(undef, layers)
  for layer = 1:layers
    weights[layer] = rand(layer_shapes[layer][1], layer_shapes[layer][2])
    biases[layer] = rand(layer_shapes[layer][1],)
  end
  z = Vector{Vector}(undef, layers)
  a = Vector{Vector}(undef, layers + 1)
  for i = 1:iterations
    num_inputs = length(inputs)
    deltas = Vector{Vector{VecOrMat}}(undef, num_inputs)
    for j = 1:num_inputs
      a, z = feed_forward(inputs[j], weights, biases)
      deltas[j] = propogate_back(y[j], a, z, weights)
      # println(deltas[j])
    end
    delta = mean(deltas)
    weights, biases = update_weights_and_biases(a, weights, biases, delta, eta)
    # if i%print_frequency == 0
    #   println("iteration $(i): error = $(sum_square_error(y, a[end]))")
    # end
  end
  return a, z, weights, biases
end

end

using .PerceptronNetwork

W1 = [0.3 0.4 0; 0 0.1 0.5]

b1 = [0.2; 0.2]

W2 = [0.6 0.2; 0.3 0.3]

b2 = [0.1; 0.1]

W3 = [0.5 0.2]

b3 = [0.4]

a0 = [2.0; 1.0; 3.0]
a1 = [3.0; 2.0; 1.0]

weights = [W1, W2, W3]

biases = [b1, b2, b3]

a, z, W, B = train_network(a0, [0.8], [(2, 3), (2, 2), (1, 2)], 0.01, 1e2)

# println("final output: $(a[end])\nfinal loss: $(loss(0.8, a[end]))")
