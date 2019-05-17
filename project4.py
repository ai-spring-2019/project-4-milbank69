"""
Nick Chkonia
Date: 5/16/2019 (with extension)
Assignment: Project 4 - Neural Network Classifiers
Instructions:

For use on different datasets, it is necessary to comment-out all other
Neural Network constructors and accuracy function calls
except the specified ones


> for use on banana.csv, generated.csv, breast-cancer-wisconsin-normalized.csv
    To initialize, use one of:
        nn = NeuralNetwork([2,2,1])
        nn = NeuralNetwork([2,1,1])
        nn = NeuralNetwork([2,6,1]
    Uncomment the same constructor in the while loop for cross-validation

> for use on increment-3-bit.csv:
    Use: nn = NeuralNetwork([3,6,3],1) and print(accuracy(nn, training,1))
                        and
    comment-out all of the lines for cross-validation in main; there will be
    comments in main to mark which lines exactly

Notes:
    - to display data in human-readable format, uncomment checkout_data in main
    - to display just the data, also uncomment "stop" to cause an error,
        useful for just seeing the input data

"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data


def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs


def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum


def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom


def accuracy(nn, pairs, threeBit = None):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    # calculating accuracy
    for (x, y) in pairs:
        nn.forward_propagate((x,y))
        class_prediction = nn.predict_class(threeBit)

        # if threebit incrementer, then we compare arrays
        if threeBit:
            if y != class_prediction:
                true_positives +=1

        else:
            if class_prediction != y[0]:
                true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction) #", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here


class Layer:
    """an abstracted array for storing which nodes are in which layer:
        makes code much more readable, as otherwise it would be a giant,
        spaghetti dish of array indexing"""

    def __init__(self, layerNumber, numNodes, startIndex, dummy = None):
        self.layer = layerNumber
        self.nodes = []
        self.populate_layer(numNodes, startIndex, dummy)


    def populate_layer(self, numNodes, startIndex, dummy = None):
        """ populate the nodes array """
        # quick modification to handle dummy layer connection generation
        if dummy:
            for _ in range(numNodes):
                self.nodes.append(startIndex)
        else:
            for i in range(startIndex, numNodes + startIndex):
                self.nodes.append(i)


class Node:
    """ each node/unit in the neural network
        ref: Russell, Norvig p.728
    """

    def __init__(self, index, layer, type, input):
        self.index = index
        self.layer = layer
        self.type = type
        self.inputVector = input
        self.inputs = []
        self.weights = []
        self.input = 0
        self.output = 0

    def displayInputs(self):
        """ repr for debugging purposes - displays inputs in inputVector"""
        for index in range(len(self.inputVector)):
            print(self.inputVector[index])


    def repr(self):
        """ more general repr for debugging purposes -
            displays info about the Node class object in human-readable
            format"""
        print("'"*100)
        print("REPR-ING")
        print("Index: {}; Layer: {}; Inputs {} ".format(self.index, self.layer,
                                        self.inputs))
        print("Inputs Vector: ", self.inputVector)
        print("Weights to self: ", self.weights)
        print("Input: ", self.input)
        print("Output:", self.output)
        print("'"*100)


    def process_input(self):
        """ sum of inputs and corresponding weights:
            essentially dotproduct"""
        self.input = 0
        for i in range(len(self.weights)):
            self.input += self.weights[i] * self.inputs[i]


    def generate_output(self):
        """ applies classification function to the output"""
        self.process_input()

        self.output = 0
        self.output = logistic(self.input)


class NeuralNetwork:
    """Neural Network Class"""
    def __init__(self, networkParamsVector):
        """
        - networkParamsVector: [<#inputLayerNodes>, ..., <#outputLayerNodes>]
        where "..." contains the number of hidden layer nodes, as many times as
        many times as there are hidden layers
        """
        self.numHiddenLayers = len(networkParamsVector) - 2
        self.numNodes = self.count_nodes(networkParamsVector)
        self.adj_matrix = []
        self.layers = []    #stores Layer class objects
        self.generate_adj_matrix(networkParamsVector)
        self.nodes = []     #stores Node class objects
        self.add_nodes()
        self.errors = []
        self.outputs = []

    def build_error_vector(self):
        """ initialize vector of errors"""
        for index in range(self.numNodes):
            self.errors.append(0)


    def count_nodes(self,nodesVector):
        """ calculates the total number of nodes"""
        totalNodes = 0
        for i in range(len(nodesVector)):
            totalNodes+=nodesVector[i]
        return totalNodes


    def print_nodes(self):
        """ repr for debugging and reference """
        for node in self.nodes:
            print("Node: {}, Index: {}, Layer: {}, Input: {}, Output {}".format(
                            node,node.index,node.layer,node.input,node.output))


    def pass_indices(self, nodeIndex, layer):
        """ helper for generate_inputs -
            passes indices """
        inputsLayer = self.layers[layer - 1].nodes
        return inputsLayer


    def generate_inputs(self, node_index, layer):
        """ helper for add_nodes -
            prepares the inputs for a given node to be added to the network,
            returns 2Darray of input indices and their associated weights"""
        inputs = []

        # if not dummy node, i.e. a hidden or output layer node
        if layer > 0 and layer < len(self.layers) - 1:
            inputs = self.pass_indices(node_index, layer)

        return inputs


    def add_nodes(self):
        """ build Node class objects and append them to self.nodes"""

        for layer in range(len(self.layers)):

            nodeIndices = self.layers[layer].nodes
            for index in nodeIndices:
                inputs = self.generate_inputs(index, layer)

                if layer == 0:
                    type = "input"
                elif layer == len(self.layers)-1:
                    type = "dummy"
                else:
                    type = ""
                self.nodes.append(Node(index, layer, type, inputs))


    def generate_weights(self, numberWeights):
        """ a nice helper for generating a weightvector of a given size"""
        weightsVector = []
        for _ in range(numberWeights):
            weightsVector.append(random.random())
        return weightsVector



    def generate_connections(self, layer_i, layer_j):
        """ Helper for generate_adj_matrix:
            - mark edges and their weights between nodes in self.adj_matrix
            IN: Layer class objects consecutive/adjacent in the neural network
        """

        i_nodes = layer_i.nodes
        j_nodes = layer_j.nodes
        numConnections = len(i_nodes) * len(j_nodes)

        # build those connections
        for i in range(len(i_nodes)):
            randomWeights = self.generate_weights(numConnections)
            node_from = i_nodes[i]
            for j in range(len(j_nodes)):
                node_to = j_nodes[j]
                thisWeight = randomWeights[j]
                self.adj_matrix[node_from][node_to] = thisWeight


    def add_dummy_node(self, networkParams):
        """ append the dummy node as a layer class object to adj. matrix,
        add connections from it to all nodes """

        # add new entries in adjacency matrix
        self.adj_matrix.append([])
        for _ in range(self.numNodes):
            self.adj_matrix[-1].append(0)

        # generate as a layer
        dummyLayer = Layer(len(networkParams), self.numNodes, self.numNodes,1)
        self.layers.append(dummyLayer)

        # generate connections between the dummy layer and all other layers
        for layer in range(len(networkParams)):
            self.generate_connections(self.layers[-1], self.layers[layer])


    def generate_adj_matrix(self, networkParams):
        """ generates the network graph, as an adjacency matrix
            - iterate pair-wise through networkParamsVector,
            generating connections
        """
        # init adjacency matrix with 0 entries
        for i in range(self.numNodes):
            self.adj_matrix.append([])
            for _ in range(self.numNodes):
                self.adj_matrix[i].append(0)

        # build up layers to know which nodes connect to which
        startIndex = 0


        for layer in range(len(networkParams)):
            thisLayer = Layer(layer, networkParams[layer], startIndex)
            self.layers.append(thisLayer)
            startIndex += networkParams[layer]

        #build up the adjacency matrix:
        for layer in range(len(networkParams)-1):
            self.generate_connections(self.layers[layer], self.layers[layer+1])

        # handle dummy node
        self.add_dummy_node(networkParams)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def return_weights(self, inputs, node_index, inputLayer = None):
        """ helper for forward_propagate -
            uses indices in inputs vector to access adjecency matrix and
            return the weights"""

        weights_vector = []
        for index in inputs:
            weights_vector.append(self.adj_matrix[index][node_index])

        # append dummy weight
        if not inputLayer:
            weights_vector.append(self.adj_matrix[-1][node_index])

        return weights_vector


    def return_inputs(self, inputs, index_j):
        """ helper to forward_propagate -
            returns vector of input values"""

        inputs_vector = []
        for input in inputs:
            inputs_vector.append(self.nodes[input].output)
        # don't forget the dummy input - always 1.0
        inputs_vector.append(1.0)

        return inputs_vector


    def forward_propagate(self, example_pair):
        """ forward propagate through the network
            ref: Norvig/Russell p.728."""

        inputs = example_pair[0]

        # assign dummy input 1.0 to dummy node(s)
        for i in range(len(self.nodes)):
            if self.nodes[i].index == self.numNodes:
                self.nodes[i].inputs = inputs[0]

        # assign remaining input values to input nodes
        for i in range(len(self.nodes)):
            node_i = self.nodes[i]

            if node_i.layer == 0:
                node_i.inputs = [inputs[i+1]]

                weight_to_i =self.return_weights([self.numNodes],node_i.index,1)
                node_i.weights = weight_to_i


                node_i.generate_output()

            else:
                break

        # forwarding through other layers
        for j in range(len(self.nodes)):
            node_j = self.nodes[j]
            nodes_i = node_j.inputVector

            #want to go until dummy nodes
            if node_j.layer == len(self.layers) -1:
                break

            # exclude input nodes
            if node_j.layer > 0:
                weights_to_j = self.return_weights(nodes_i, node_j.index)
                node_j.weights = weights_to_j
                inputVals = self.return_inputs(nodes_i, node_j.index)
                node_j.inputs = inputVals
                node_j.generate_output()


    def deltas_backward_output(self, index, y_val):
        """ helper for back_propagation_learning -
            propagate deltas backward for output layer nodes
            from class:
                (1) delta[j] = g(in_j) * (1-g(in_j)) * (y_j - a_j)
                                    OR
                (2) delta[j] = a_j * (1-a_j) * (y_j - a_j)
            """

        # calculate error
        output = self.nodes[index].output
        input = self.nodes[index].input
        new_val = logistic(input) * (1-logistic(input)) * (y_val[0] - output)

        self.errors[index] = new_val


    def weights_on_errors(self, index_i, weights):
        """ helper for deltas_backward_others -
            computes sum_j((w_{i,j} * delta[j])) """

        sum = dot_product(self.errors, weights)
        return sum


    def deltas_backward_others(self, index_i):
        """ helper for back_propagation_learning -
            propagate deltas for other layer nodes
            from class:
                delta[i] = g(in_i) * (1 - g(in_i)) * sum_j((w_{i,j} * delta[j]))"""

        weights_i = self.adj_matrix[index_i]
        input = self.nodes[index_i].input
        sum_errors = self.weights_on_errors(index_i, weights_i)
        new_val = logistic(input) * (1 - logistic(input)) * sum_errors
        self.errors[index_i] = new_val


    def back_propagation_learning(self, examples):
        """ back propagate to learn the weights
            ref: Norvig/Russell p.734. """

        self.build_error_vector()
        epochs = 1000
        for epoch in range(epochs):

            # alpha value used in weight-learning
            alpha = 1000 / 1000 + epoch
            for pair in examples:
                # do forward propagation portion
                self.forward_propagate(pair)

                # now do back propagation

                # generate errors on output layer
                for j in range(len(self.nodes)):
                    j_node = self.nodes[j]
                    # back propagate errors on output layer
                    if j_node.layer == len(self.layers) - 2:
                        # self.update_error_value(j_node.index, "back")
                        self.deltas_backward_output(j_node.index,pair[1])

                # back-propagate errors on other layers, R->L
                for l in range(len(self.layers)-3, -1, -1):
                    layer_l = self.layers[l]
                    for node_i in range(len(layer_l.nodes)):
                        node_index = layer_l.nodes[node_i]
                        self.deltas_backward_others(node_index)

                # update every weight in network using errors
                for i in range(len(self.adj_matrix)):
                    node_i = self.adj_matrix[i]
                    for j in range(len(node_i)):
                        w_ij = node_i[j]
                        if w_ij != 0:
                            new_val = alpha * self.nodes[i].input * self.errors[j]
                            w_ij += new_val
                            if math.isnan(w_ij): balls
                            self.adj_matrix[i][j] = w_ij

                self.update_outputs()


    def data_outputs(self, pairs):
        """ debugger helper function, print outputs on data pairs passed-in"""
        outputs = []
        for i in range(len(pairs)):
            outputs.append(pairs[i][1])
        return outputs


    def predict_class(self, threeBit):
        """ predicts the class (y value) based on x_values -
            rounds to nearest int
        """

        # multiclass style classification here
        if threeBit:
            output_val = []
            output_layer = self.layers[-2]
            for i in output_layer.nodes:
                value = self.nodes[i].output
                value = round(value)
                output_val.append(value)
        else:
            output_val = self.nodes[self.numNodes-1].output
            output_val = round(output_val)

        return output_val


    def update_outputs(self):
        """ helper for get_outputs-
            appends current output to list of outputs for neural network"""
        output_layer = self.layers[-2]
        for index in output_layer.nodes:
            output = self.nodes[index].output

            self.outputs.append(output)


    def get_outputs(self):
        """ returns outputs of network in an array"""
        return self.outputs


### Neural Network code ends here
################################################################################


def checkout_data(data):
    """ helper for main -
        in-case user wants to view all tuples in data"""
    for datum in data:
        print(datum,"\n")


def preprocess_data(data, subset_size, num_subsets, start_index, end):
    """ preprocess input data for cross_validation
        returns tuples of data in range(start_index,start_index+subset_size)"""

    subset = []
    for i in range(end):
        subset.append(data[i])

    return subset


def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data, if you want
    checkout_data(training)

    # useful to sometimes just view the data-
    # uncomment below and the interpreter will return an error
    # stop


    # initialize neural network here.........................................
    # use for 3-bit incrementer
    # nn = NeuralNetwork([3, 6, 3])

    # use for others
    nn = NeuralNetwork([2,2,1])
    # nn = NeuralNetwork([2,1,1])
    # nn = NeuralNetwork([2,6,1]
    #  .....................................................................

    # uncomment these  for 3-bit incrementer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # nn.back_propagation_learning(training)
    # print(accuracy(nn, training,1))
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # cross_validation implemented below xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # comment it out for 3-bit incrementer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # run with cross_validation, 5 subsets for simplicity
    num_subsets = 5
    subset_size = len(training) / num_subsets
    start_index = 0

    accuracies = []


    # comment this out for 3-bit incrementer ----------------------------------
    # run k iterations
    # for i in range(num_subsets):

    while start_index  < len(training):
        # need to re-initialize neural network every iteration to clear values
        nn = NeuralNetwork([2,2,1])
        accuracy_value = 0
        start = int(start_index)
        end = start + subset_size
        end = int(end)

        test_data = preprocess_data(training, subset_size, num_subsets,
                                                    start, end)
        train_data = training[0:start] + training[end:]

        nn.back_propagation_learning(train_data)
        accuracy_value = accuracy(nn, test_data)
        accuracies.append(accuracy_value)
        # update starting index for preprocessing for next iteration
        start_index += subset_size

    # optional, can comment back-in to see accuracies for fun
    # print(accuracies)
    average_accuracy = 0
    # compute and return the average
    for accuracy_value in range(len(accuracies)):
        average_accuracy += accuracies[accuracy_value]

    average_accuracy = average_accuracy / len(accuracies)
    print(average_accuracy)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


if __name__ == "__main__":
    main()
