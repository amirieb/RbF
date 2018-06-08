# RbF

This code demonstrates how RbF (see below) can be used to train any
neural network. RbF is inspired by the broad evidence in psychology
that shows human ability to retain information improves with repeated 
exposure and exponentially decays with delay since last exposure. It 
works based on spaced repetition in which training instances are 
repeatedly presented to the network on a schedule determined by a spaced 
repetition algorithm. RbF shortens or lengthens review intervals for 
training instances with respect to loss of instances and current 
performance of network on validation data.

To use this code, you just need to load your data (lines 47-56), design
your favorite network architecture (lines 61-67), and choose the type of
training paradigm you'd like to use (lines 77-83 for standard training 
and lines 88-96 for Rbf). If you are using only one of these training
paradigms, edit/comment out lines 101-107.  

https://scholar.harvard.edu/hadi/RbF 
Please see the above address for most recent updates on RbF. 

# How to Use
python rbf.py


### Parameters 
**kern**: type of kernel function, it could be any kernel in ['gau', 'lap', 'lin', 'cos', 'qua', 'sec'] which represent gaussian, laplace, linear, cosine, quadratic, and secant functions respectively
            
**nu**: recall confidence, RbF scheduler estimates the maximum delay such that instances can be recalled with this confidence in the future iterations, nu takes a value in (0,1)  


# Citation
Hadi Amiri, Timothy A. Miller, Guergana Savova. [Repeat before Forgetting: Spaced Repetition for Efficient and Effective Training of Neural Networks](http://aclweb.org/anthology/D17-1255). EMNLP 2017. 

# Contact
Hadi Amiri, hadi.amiri@childrens.harvard.edu
