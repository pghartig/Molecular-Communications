# Communication System Project
Welcome! This project can thought of a two sub-projects:
1.  [A framework for development and testing of communication system components](framework.md)
2.  [Various component implementations utilizing the above framework](implementations.md)

If you are interested in using the framework to develop system components, please read the [usage tutorial](tutorial.md).
The table of contents in the [implementations](implementations.md) section provides a list and brief description of some of the components implemented up to now.




# Molecular Communications Project

The objective of this project is to evaluate the performance of decoding algorithms which use neural networks in molecular
communications channels. In particular, this work will focus on incorporating neural networks into the Viterbi algorithm decoding
framework. As the Viterbi algorithm is ideal when knowledge of the channel model and the current model parameters are known, this will
provide an environment in which to benchmark the performance of a neural network based approach. Further, for many channels in molecular
communications, models are not available or finding parameters of the model may be difficult. Incorporating neural networks into this
framework allows for models to be created in a data-driven approach.
