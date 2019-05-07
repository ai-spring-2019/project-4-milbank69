"""
Nick Chkonia
Date:
Usage: python3 project3.py DATASET.csv
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

def accuracy(nn, pairs):
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

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

# NOTE: need to know edges and their weights: which nodes are connected and to what degree
class Node:
    """ each node/unit in the neural network"""

    def __init__(self,index,layer,inputs):
        # init with unique identifiers
        self.inputsVector = inputs
        self.index = index
        self.layer = layer
        self.output = self.generate_output()

    def generate_output(self):
        pass


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
        self.generate_adj_matrix(networkParamsVector)


    def count_nodes(self,nodesVector):
        """ calculates the total number of nodes"""
        totalNodes = 0
        for i in range(len(nodesVector)):
            totalNodes+=nodesVector[i]
        return totalNodes


    def generate_weights(self, numberWeights):
        """ a nice helper for generating a weightvector of a given size"""
        weightsVector = []
        for _ in range(numberWeights):
            weightsVector.append(random.random())
        return weightsVector
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   NOTE: How can I use the layer number and the size of each layer to inform
          # me of which index I am starting inside of the adjacency matrix?
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def generate_connections(self, layer_i, layer_j):
        """ Helper for generate_adj_matrix:
            - mark connections between layers on self.adj_matrix
        """
        print("generate_connections")
        print(layer_i, layer_j)
        print(self.adj_matrix)
        numConnections = layer_i * layer_j
        # tricky indexing, needed for appropriate adjacency marking
        offset = layer_i

        randomWeights = self.generate_weights(numConnections)
        print("randomWeights: ", randomWeights)
        print("numConnections: ", numConnections)

        for i in range(layer_i):
            for j in range(layer_j):
                self.adj_matrix[i][offset+j] = randomWeights[j]
        print("after connecting:", self.adj_matrix)
        print("\n")

        # for row in range(layer_i):
            # self.adj_matrix[row].append(randomWeights[row])

    def generate_adj_matrix(self, networkParams):
        """ generates the network graph, as an adjacency matrix
            - iterate pair-wise through networkParamsVector,
            generating connections
        """
        # build-up empty rows
        for i in range(self.numNodes):
            self.adj_matrix.append([])
            for _ in range(self.numNodes):
                self.adj_matrix[i].append(0)

        # populate the matrix
        for layer in range(len(networkParams)-1):
            self.generate_connections(networkParams[layer],
                                            networkParams[layer+1])
        print("final adjacency matrix: \n", self.adj_matrix)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def forward_propagate(self, input):
        """ forward propagate through the network. on some input"""
        pass


    def back_propagation_learning(self,examples):
        """ back propagate to learn the weights"""
        pass


    def cross_validation(self):
        """ cross-validation to predict accuracy"""
        pass



### Neural Network code ends here
################################################################################



def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in training:
        print(example)

    print("\n")


    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    nn = NeuralNetwork([2,2,1])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
