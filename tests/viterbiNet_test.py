import torch
import numpy as np
from communication_util import training_data_generator


def test_viterbi_net():
    """
    Testing the setup and training of a neural network using the viterbiNet Architecture
    :return:
    """

    """
    Choose platform
    """
    device = torch.device('cpu')
    # device = torch.device('cuda') # Uncomment this to run on GPU

    """
    Setup Training Data
    """
    number_symbols = 60

    channel = 1j * np.zeros((1, 5))
    channel[0, [0, 3, 4]] = 1, 0.5, 0.4
    data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols+2*channel.size),
                                       SNR=10, plot=True, channel=channel)
    data_gen.setup_channel(shape=None)
    data_gen.random_symbol_stream()
    data_gen.send_through_channel()

    x, y = data_gen.get_labeled_data()
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    """
    Setup Architecture
    """
    m = data_gen.alphabet.size
    channel_length = data_gen.CIR_matrix.shape[1]



    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = number_symbols, 1, 100, 50, np.power(m, channel_length)

    # Create random Tensors to hold input and outputs
    # x = torch.randn(N, D_in, device=device)
    # y = torch.randn(N, D_out, device=device)


    # Create random Tensors for weights; setting requires_grad=True means that we
    # want to compute gradients for these Tensors during the backward pass.
    # w1 = torch.randn(D_in, H1, device=device, requires_grad=True)
    # w2 = torch.randn(H1, H2, device=device, requires_grad=True)
    # w3 = torch.randn(H2, D_out, device=device, requires_grad=True)
    w1 = torch.randn(D_in, H1, device=device, requires_grad=False)
    w2 = torch.randn(H1, H2, device=device, requires_grad=False)
    w3 = torch.randn(H2, D_out, device=device, requires_grad=False)

    learning_rate = 1e-6
    for t in range(50):
        # Forward pass: compute predicted y
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)

        h2 = h1.mm(w2)
        h2_relu = h2.clamp(min=0)

        y_pred = h2_relu.mm(w3)

        # Compute and print loss; loss is a scalar, and is stored in a PyTorch Tensor
        # of shape (); we can get its value as a Python number with loss.item().
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w3 = h2_relu.t().mm(grad_y_pred)

        grad_h2_relu = grad_y_pred.mm(w3.t())
        grad_h2 =grad_h2_relu.clone()
        grad_h2[h2 < 0] = 0
        grad_w2 = h1_relu.t().mm(grad_h2)

        grad_h1_relu = grad_h2_relu.mm(w2.t())
        grad_h1 = grad_h1_relu.clone()
        grad_h1[h1 < 0] = 0
        grad_w1 = x.t().mm(grad_h1)


        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3


    # Make sure to output these trained weights so that they can be used without training again
    torch.save([w1, w2, w3], 'weights.pt')
    assert False
