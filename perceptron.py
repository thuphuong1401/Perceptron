#!/usr/bin/python3

# AUTHOR:  *Phuong Vu*
# NetID:   *pvu3 (e.g., blackboard)
# csugID:  *your csug login here (if different from NetID*

import numpy as np
# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points

# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
# Here +1 means add bias
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        return xs, ys

# Do learning.
def perceptron(train_xs, train_ys, iterations, filename):
    train_xs = np.asarray(train_xs)
    train_ys = np.asarray(train_ys)

    # [1] because of numbers of features of x
    weights = np.zeros(train_xs.shape[1])
    num_iter = 0

    with open(filename, 'w') as file:
        file.write("Number of Iterations, Accuracy \n")
        while num_iter < iterations:
            for x,y in zip(train_xs, train_ys):
                y_pred = np.sign(np.dot(np.transpose(weights), x)) # wTx > 0 or wTx < 0
                if y_pred != y:
                    weights = np.add(weights, np.array(y*x))
                    #print(test_accuracy(weights, train_xs, train_ys))

            # Record number of iterations and corresponding accuracy into a file
            # Write sparsely to save space
            if num_iter%20 == 0:
                file.write("{}, {} \n".format(num_iter, test_accuracy(weights, train_xs, train_ys)))

            num_iter += 1

        # perfect accuracy, then just break
            if test_accuracy(weights, train_xs, train_ys) == 1.0:
                print("The perceptron algorithm halts at the {}th iteration.".format(num_iter))
                break

    return weights

# Calculate R as in the perceptron convergence theorem
def calculate_R(train_xs):
    train_xs = np.asarray(train_xs)
    mag_array = np.empty(0)
    # Calculate magnitude of each feature vector
    for row in train_xs:
        magnitude = 0.0
        for col in row:
            magnitude += col**2

        magnitude = magnitude**0.5
        mag_array = np.append(mag_array, [magnitude])
        #print(mag_array.shape)

    #print(mag_array)
    return np.max(mag_array)



# Return the accuracy over the data using current weights.
def test_accuracy(weights, test_xs, test_ys):
    test_xs = np.asarray(test_xs)
    test_ys = np.asarray(test_ys)
    test_y_pred = [np.sign(np.dot(weights,x)) for x in test_xs]
    # Compute the classification accuracy in the test set
    accuracy = sum(test_y_pred == test_ys)/len(test_ys)
    # Round the accuracy to 8 decimals
    accuracy = round(accuracy,8)
    return accuracy


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=200000, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default='data/challenge0.dat', help='Training data file.')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(args.train_file)
    print(calculate_R(train_xs))

    name = args.train_file.split('.')
    file_name = name[0] + ".txt"
    weights = perceptron(train_xs, train_ys, args.iterations, file_name)
    accuracy = test_accuracy(weights, train_xs, train_ys)
    print('Train accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

if __name__ == '__main__':
    main()
