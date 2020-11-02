import numpy as np




    ########################## Sigmoid Activation Function ###############################

def sigmoid(x):
        """
        Computes the sigmoid function of the inputs received by the unit;
        ||| sigm_fun(x)= 1/[1+exp(-x)] |||

        :param x: net -> input's weighted sum
        :return: output of the unit
        """
    return 1. / (1. + np.exp(-x))

def sigmoid_deriv(x):
        """
        Computes the derivative of the sigmoid function;
        ||| sigm_fun(x)' = sigm_fun*(1-sigm_fun) |||

        :param x: net -> input's weighted sum
        :return: derivative of the sigmoid function
        """
    return np.diag(sigmoid(x) * (1 - sigmoid(x)))

    ############################# SoftMax Activation Function ###########################

def softmax(x):
        """
        Computes the softmax function of the input received by the unit:
        ||| softmax_fun(x) = exp(x - max(x)) |||

        :param x: net -> input's weighted sum
        :return: output of the unit
        """
    return np.exp(x - np.max(x))

def softmax_deriv(x): #TODO: to be completed
        """
        Computes the derivative of the softmax function:
        ||| softmax_fun(x)' = to be completed **** |||

        :param x: net -> input's weighted sum
        :return: derivative of the softmax function
        """
    softmax_fun = np.exp(x - np.max(x))

    ############################# ReLU Activation Function #################################
def relu(x):
        """
        Computes the ReLU function:
        ||| relu_fun(x) = max(0,x) |||

        :param x: net -> input's weighted sum
        :return: output of the unit
        """
    return np.maximum(0, x)

def relu_deriv(x):
        """
        Computes the derivative of the ReLU function:
        |||  relu_fun(x)' = 0  if x<0   |||
        |||  relu_fun(x)' = 1  if x>=0  |||

        :param x: net-> input's weighted sum
        :return: derivative of the ReLU function
        """
    if x < 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    # TODO: aggiungi test qui, tipo print(sigmoid(5)) e vedi se il valore è giusto
