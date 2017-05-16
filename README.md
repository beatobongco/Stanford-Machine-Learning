# Stanford Machine Learning

Notes and code for Andrew Ng's class on machine learning.

## Introduction

What is Machine Learning?

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

Supervised learning - predictive. (1) regression - predict a value we haven't seen yet (2) classification - classify something (as if into a category)

Unsupervised - find underlying structure of data. Clustering. Non-clustering - find structure in chaos, like separating voices in audio.

## Model representation

Conventions

```
m - number of training examples
x - input var / features
y - output var / target
(x, y) - training example
xi - ith training example's x
h - hypothesis function, function(x) that outputs the estimated val of y. Maps from x's to y's. example: hθ(x) = θ0 + θ1x
theta (θ) - just means an unknown that we're trying to find using ML.
```

minimize θ0 θ1 - we're minimizing the sum of squared error (see notes on [Curve Fitting](https://github.com/beatobongco/TIL/blob/master/day_notes/2017-05-04_Curve_Fitting.md)) / 2m.

J(θ0, θ1) - squared error cost function, used to measure the accuracy of our hypothesis function

Why do we square errors? Most commonly used one for regression problems.

The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the term.

