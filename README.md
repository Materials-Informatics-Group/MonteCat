# MonteCat
The MonteCat Code

This script follows the proposed MonteCat algorithm that constructs a Regression Model from a big pool of engineered Descriptors (Features) through an adaptation of the Metropolis-Hastings algorithm. The number of iterations and the Temperature modulating the Acceptance Probability are determined by the user.

The MonteCat code was written in Python3 and uses the NumPy, pandas and scikit-learn libraries for its calculations. This script only needs an input dataset (the training data) and outputs a report file continuously overwritten in real time detailing the outcomes of each iteration of the algorithm, as well as a filtered training data containing only the descriptor variables at the end of the code's execution and the target variable.
