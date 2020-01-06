# Communication System Project
Welcome! This project can thought of a two sub-projects:
1.  [A frameworkfor development and testing of communication system components](framework.md)
2.  [Various component implementations utilizing the framework](implementations.md)

If you are interested in using the framework to develop system components, please read the [usage tutorial](tutorial.md).
The table of contents in [implementations](implementations.md) section provides a list and brief description
of some of the components implemented up to now.


1.  Pulse shaping and reciever filtering
2.  Equalization and decoding
3.  Channel Coding schemes


## Structure
The framework is centered around the a class (docs/data_gen.md) used for configuring communication system parameters including:
1.  Number or receivers/transmitters
2.  Symbol constellation and pulse shape
3.  Communication channel
4.  Filtering and sampling at receiver


# Molecular Communications Project

The objective of this project is to evaluate the performance of decoding algorithms which use neural networks in molecular
communications channels. In particular, this work will focus on incorporating neural networks into the Viterbi algorithm decoding
framework. As the Viterbi algorithm is ideal when knowledge of the channel model and the current model parameters are known, this will
provide an environment in which to benchmark the performance of a neural network based approach. Further, for many channels in molecular
communications, models are not available or finding parameters of the model may be difficult. Incorporating neural networks into this
framework allows for models to be created in a data-driven approach.
