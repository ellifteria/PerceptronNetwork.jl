module NetworkIO

using Serialization

export read_network, write_network

function write_network(biases, weights, file_path)
  data_to_write = [biases, weights]
  serialize(file_path, data_to_write)
end

function read_network(file_path)
  data_read = deserialize(file_path)
  return data_read[1], data_read[2]
end

end
