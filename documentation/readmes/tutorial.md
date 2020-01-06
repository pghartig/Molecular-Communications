# Usage Tutorial
This tutorial will walk through an example of using the development framework described in the [introduction](framework.md) to implement a couple of basic components of a communication system.

##  Outline
The end goal of this example will be to implement a basic  channel coding scheme and detector for a basic communication channel. The steps will be:

1.  Describe a transmit signal
2.  Encode the transmit signal with the channel coding scheme
3.  Transmit the encoded signal
4.  Perform hard-decision detection on the received signal
5.  Decode the received detected signal using the channel code

##  Implement
1. Describe a transmit signal
  *   First we generate an information sequence to be transmitted and transform this into the chosen symbol constellation.

  ```python
  code
  ```
  * Next, we decide on the number of transmitted streams. In this case we will use a single input stream, single output stream transmission channel (SISO). We will also choose a Signal to Noise Ratio for the channel (AWGN is default noise type at receiver).
  ```python
  SNR=5
  number_symbols = 1000
  channel = np.zeros((1, 5))
  # channel[0, [0, 1, 2, 3, 4]] = 1, .1, .01, .1, .04
  channel[0, [0, 1, 2, 3, 4]] = 1, .1, .1, .1, .4
  data_gen = training_data_generator(symbol_stream_shape=(1, number_symbols), SNR=SNR, plot=True, channel=channel)
```
2.  We now encode this information symbol stream using a basic channel code in which each symbol is repeated 3 times.
  *
  ```python
  code
  ```

3.  The encoded symbol stream is now sent through the channel. Note that this signal could optionally be modulated onto a continuous time pulse before transmission but for this purpose of this tutorial we transmit the symbols through a linear channel.


4.  Using the received signal, we now apply a hard-decision detection scheme.

5. Finally, we find a detected information sequence using the rules of the channel coding scheme.
