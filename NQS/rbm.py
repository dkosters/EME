import torch
import torch.nn as nn
import numpy as np
import time
import argparse


class RBM(nn.Module):
    def __init__(self, n, alpha):
        super(RBM, self).__init__()
        self.l1 = nn.Linear(n, n*alpha)

    def forward(self, x):
        y = self.l1(x)
        y = torch.exp(torch.sum(torch.log(2 * torch.cosh(y)), 1))


def normalize(y):
    return torch.div(y, torch.norm(y, p=2))

def dec_to_bin(x, y):  # converts a digit to a binary with y entrees
    return format(x, "0{0}b".format(y))

def neutral_states(n):  # gives all spin convigurations of n spins with total spin 0
    states = np.zeros([2**n, n], dtype=np.float64)
    for i in range(0, 2**n, 1):
        for j in range(0, n, 1):
            states[i, j] = 2 * int(dec_to_bin(i, n)[j]) - 1
    neutral_states = np.empty([0, n], dtype=np.float64)
    for i in range(0, 2**n, 1):
        if sum(states[i, :]) == 0:
            neutral_states = np.vstack((neutral_states, states[i,:]))
    return neutral_states

def main(FLAGS):

    load_time_start = time.time()
    print('start loading settings')

    # precission 
    if FLAGS.dtype == "fp64":
        torch.set_default_dtype(torch.float64)
    elif FLAGS.dtype == "fp32":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float16)
    
    # print precission 
    torch.set_printoptions(precision=12)

    alpha = FLAGS.alpha  # ratio between input and first layer layer density 
    iterations = FLAGS.iterations  # number of times the entire inference precedure is repeated 
    device = torch.device(FLAGS.device)  # the computing device eather CPU or GPU 

    n = 16  # number of spins

    print("start loading weights")
    # read files
    with open('optimized_W/alpha={0}/b.csv'.format(alpha), 'r') as file:
        biases = np.loadtxt(file, delimiter=",", dtype=np.float64)

    with open('optimized_W/alpha={0}/W.csv'.format(alpha), 'r') as file:
        weights = np.loadtxt(file, delimiter=",", dtype=np.float64).transpose()

    with open('ED_amplitude.csv', 'r') as file:
        true_wf = np.loadtxt(file, delimiter=",", dtype=np.float64)

    # setup network 
    print('Setup network on device')
    rbm = RBM(n, alpha).to(device)

    # set weights 
    sd = rbm.state_dict()
    sd['l1.weight'] = torch.Tensor(weights)
    sd['l1.bias'] = torch.Tensor(biases)
    rbm.load_state_dict(sd)

    # send input to device 
    states = torch.Tensor(neutral_states(n)).to(device)

    load_time_end = time.time()
    print("total loading time: {0} s".format(load_time_end-load_time_start))

    # set modol in evaluation mode 
    rbm.eval()

    # inference 
    with torch.no_grad():
        for i in range(0, iterations, 1):
            time1 = time.time()
            output = rbm.forward(states)
            print("iteration{0} time: {1}s".format(i, time.time() - time1))

    inference_time_end = time.time()


    print("alpha: {0}, iterations: {1}, device: {2}".format(alpha, iterations, device))
    print('load time: {0}s, inference time: {1}s'.format(load_time_end - load_time_start, inference_time_end - load_time_end))
    accuracy_calc_start = time.time()
    output = output.to('cpu')
    output = normalize(output)
    print('accuracy: {0}'.format(torch.sum(torch.mul(output, torch.Tensor(true_wf)))))
    print('load time: {0}s, inference time: {1}s, accuracy computation time: {2}s'.format(load_time_end - load_time_start, inference_time_end - load_time_end, time.time() - accuracy_calc_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--iterations', type=int, default=100000)

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)