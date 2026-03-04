# Fully Analog Resonant Recurrent Neural Network (R<sup>2</sup>NN)
## Overview
The metacircuit-based R<sup>2</sup>NN is realized by first training a R<sup>2</sup>NN on a designated dataset using a digital training framework to achieve accurate classification of signal categories. Once the iteration meets the predefined criterion, the parameters in the R<sup>2</sup>NN are then converted into corresponding metacircuit elements using the reformulated MEA. This parameter transformation yields a functionally equivalent metacircuit-based R<sup>2</sup>NN. 

## Acknowledgements
We would like to express our sincere gratitude to the authors of the **[WaveTorch](https://github.com/fancompute/wavetorch.git)** project. Our codebase is developed based on the pioneering work of `wavetorch`. We have utilized their foundational framework for wave propagation simulation and gradient-based optimization in physical systems, extending it to support R<sup>2</sup>NN dynamics. This project would not have been possible without their open-source contribution to the community.
