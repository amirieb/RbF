# RbF

This code demonstrates how RbF (see below) can be used to train any
neural network. RbF is inspired by the broad evidence in psychology
that shows human ability to retain information improves with repeated 
exposure and exponentially decays with delay since last exposure. It 
works based on spaced repetition in which training instances are 
repeatedly presented to the network on a schedule determined by a spaced 
repetition algorithm. RbF shorten or lengthen review intervals for 
training instances with respect to loss of instances and current 
performance of network on validation data.

To use this code, you just need to load your data (lines 47-56), design
your favorite network architecture (lines 61-67), and choose the type of
training paradigm you'd like to use (lines 77-83 for standard training 
and lines 88-96 for Rbf). If you are using only one of these training
paradigms, edit/comment out lines 101-107.  

https://scholar.harvard.edu/hadi/RbF 
Please see the above address for most recent update on RbF. 

Citation
Amiri, et al., Repeat before Forgetting: Spaced Repetition for Efficient 
and Effective Training of Neural Networks. EMNLP 2017.

Contact
Hadi Amiri, hadi.amiri@childrens.harvard.edu