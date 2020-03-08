# Communication System Component Development and Testing framework

The goal of this framework is to provide a consistent communication system structure that may be used during the development of new system components such as:

1.  Pulse shaping and receiver filtering
2.  Equalization and decoding
3.  Channel Coding schemes


## Structure
The framework is centered around a class (docs/data_gen.md) used for configuring basic communication system parameters including:
1.  Number or receivers/transmitters
2.  Symbol constellation and pulse shape
3.  The communication channel(s)
4.  Filtering and sampling at receiver
