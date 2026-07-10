---
title: 'An introduction to Deep Adaptive Design (DAD)'
date: 2026-07-10
permalink: /posts/2026/07/DAD/
tags:
  - BED
  - ML
author_profile: false
---

In the blog post, I introduce [Deep Adaptive Desing (DAD)](https://arxiv.org/pdf/2103.02438), an increasingly common computational appraoch to implement Bayesian Experimental Design. The post includes a whistle stop tour of the core concepts of BED, Neural Networks and Embeddings, written with statisticians in mind who are new to the world of Machine Learning (like me)!

## A whilstle stop tour of Bayesian Experimental Design 
Bayesian Experimental Design (BED) formalises what it means for an experiment to be optimal. If you were running a sequential experiment, it would be human nature to choose the experiment based on how relevant its outcome could be to the task at hand and on where our uncertainty is greatest. In essence, Bayesian Experimental Design replicates this decision-making. Though, if we wish to write this decision-making down, beliefs are informed by priors on statistical models and uncertainty is statistically propagated across our uncertainties in model parameters and potential outcomes. 

One of the most common approaches to conducting an optimal experiment is by considering an experiment's expected information gain (EIG). An experiment that maximises EIG is expected to produce the strongest update to our beliefs when we observe its outcome.

I like to explain BED in terms of Wordle. All Wordle users have the same goal - to identify the word of the day given as few guesses as possible. We wish to **learn** what the secret word of the day is, which in statistical terms can be translated to learning about an unknown parameter $\theta$ (or word in this case). As Wordle players, we are in control of the designs $$\xi$$ we choose  which, in this case, represent the potential 5-letter word choices we can make. Once we have guessed a word, we observe an outcome $$y$$ and then guess a new five-letter word using this guess. For Worldle, $$y$$ is the coloured code representing how many letters of the secret word we correctly identified and their location in the word. 

At the very start of playing Wordle, we wish to choose the next word which maximises the EIG across all potential 5-letter words $$\xi \in \Xi$$ we can guess. The EIG for each word at the start of WORDLE can be written as, 

$$
EIG_\theta(\xi) = \int_{y \in \mathcal{Y}} p(y \mid \xi) \int_{\theta \in \Theta} p(\theta \mid y, \xi) \log \frac{p(\theta \mid y, \xi)}{p(\theta)} d\theta dy. \qquad (1)
$$

To understand this formula, suppose we wish to quantitatively evaluate the benefit of guessing the word 'ADIEU' for today's Wordle. 

**Inner integral**:
Suppose we guessed 'ADIEU' and observed outcome ⬜⬜⬜🟨⬜. In this case, we know only the letter E from ADIEU is in the secret word, though it is in the wrong space as it stands.To evaluate how good our guess of 'ADIEU' was having observed outcome ⬜⬜⬜🟨⬜, we need to evaluate how much we learnt across all potential secret words $\theta \in \Theta$ (around 2,300 options). Thus, we first propagate our uncertainty across all potential parameter options $\theta$ we may see. For example, this guess certainly means we no longer suspect 'BUZZY' of being the secret word, but it hasn't helped us learn whether the secret word is either 'WHOLE' or 'PROSE'. This inner integral helps us mathematically quantify this learning. 

**Inner integral**:
 The outer integral recognises that we cannot observe outcome ⬜⬜⬜🟨⬜ without choosing ADIEU as a guess to start with. This goes against the whole point of the challenge, we want to know if 'ADIEU' is a good word before wasting a Wordle turn on it. In this case we must also propagate our uncertainty once more over all the potential colour patterns we may see for the word 'ADIEU'. For example, as 'ADIEU' has lots of vowels, we expect the colour pattern to have more yellows and greens that a more unusual word like 'FLUFF' which, with its repeated F's, is less likely to share many letters with the secret word. 

Accordingly, the best first guess for Wordle will be $$\xi^* = \arg \max_{\xi \in \Xi} EIG_\theta(\xi).$$

Whilst it is not notationally challenging to write the objective for the best Wordle word, computationally it is a different story. Maximising the EIG is a doubly intractable objective, which introduces a huge computational bottleneck for the real time deployment of experiments wishing to leverage BED. This sets the scene for DAD (Deep Amortised Design).

## Amortising BED

The idea of *amortising* processes has come up many times since starting my postdoc. In the previous section we discussed the case where at each step of an experiment, a doubly intractable objective must be optimised, with a huge computational cost. *Amoritisation* considers an approach where rather than case evaluating this objective at each data acquisition step, we instead train a strategy upfront, before *deployment* (active use). By doing so, model training is completed before the experiment begins, ensuring that decisions can be made almost instantaneously once the experiment is underway.

For the DAD, this strategy means that a *Design policy* $\pi$ (Neural Network) is trained to acquire data intelligently before the experiment so that, once the experiment commences, decision-making is as simple of completing one *forward pass* (one model fit) of the network to recommend the next experiment. 

As the neural network is trained before the experiment has even been run, the neural network is trained on synthetic data using the known prior and likelihood of the model. Then, after model training, the design policy can be used in real time by practitioners for almost instantaneous, real-time for decision-making, bypassing the computationally expensive necessity to compute the doubly intractable objective (1) to compute the EIG. 

## Neural Networks 

Whilst neural Networks have always struck me as mystical black box, broadly speaking their specific approach to model fitting is somewhat similar to more statistically-minded estimators such as ordinary least squares estimator for linear regression. It is more so their intricate architecture rather than their broad methodology which differs.

Like any statsitical model, with a neural network you have a number of inputs which you wish to use to predict some output. Similarly to a linear regression, a neural network takes these inputs and finds a relationship to map these inputs to an output. The exact relationship between inputs and outputs is formed in hidden layers of the neural network. 

Consider a simple neural network with $L$ layers which looks something like,

IMAGE 

We start with our raw inputs $x$. These raw inputs are linearly transformed given a weight $$w^{(1)}$$ and bias $$b^{(1)}$$ such that $$z^{(1)} = w^{(1)}x + b^{(1)}$$. $$z^{(1)}$$ is then transformed using non-linear transformation $$\sigma^{(1)}$$ such that $$a^{(1)} = \sigma^{(1)}(z^{(1)})$$. Exemplar functions $$\sigma^{(1)}$$ can include sigmoid() or tanh() functions. 

Now at the second layer, the activation function consists of a non-linear transformation of a linear transformation of the proceeding activation function. For example, $$a^{(2)} = \sigma^{(2)}(w^{(2)}a^{(1)} + b^{(2)}) = \sigma^{(2)}(w^{(2)}\sigma^{(1)}(w^{(1)}x + b^{(1)}) + b^{(2)})$$. 
A third layer has an activation function which is function of the proceeding activation function from the second layer and so on. By adding more layers to the neural network, we increase its *width*. Once we have computed the activation function for the final layer $L$, there are by now many parameters (weights and biases) associated with the predictive task. 

Now we have to evaluate the predictive performance of the Neural network. To do so, when training the neural network, we compare our network's prediction to the true outcome using a *loss*, such as the sum of squared residuals used in linear regression. 

A perfect model would have zero loss and as such, we wish to train our neural network to minimise the loss as best as possible. Accordingly, during training, we iteratively tune the neural network's parameters (weights and biases) in the hope that by tuning these parameters the network will perform better at a future predictive task. We do this by *backpropagating* the neural network. This just means we differentiate the loss with respect to all the parameters. Given the graphical nature of the network, with all weights and biases nested in the final activation function, this can be done using the chain rule. By differentiating, we can identify the rate at which each parameter contributes to the overall loss and as such weights and biases can be tuned to reduce this loss and thus improve model prediction. Once the network has been backpropogated and gradients for each parameter identified, the network can be *optimised*. This process of making a *forward pass* of the model (running the model for new input data), comparing the prediction to the true outcome using a loss, backpropogating, and optimising the network is repeated over and over again using training data until the model is sufficiently trained and ready for deployment.

Whilst this section has introduced neural networks considering an example with a single node per layer, we can increase the *depth* of the network by adding multiple nodes per layer. Intuitively, this does not add much additional complication with the only change being that the activation for each node is now an aggregate of the weights and biases associated with the nodes in the previous layer. 

```python 
import torch 
import torch.nn as nn
import os
```
