#A tour through tensorflow with financial data

I present several models ranging in complexity from simple regression to LSTM and policy networks. The series can be used as an educational resource for tensorflow or deep learning, a reference aid, or a source of ideas on how to apply deep learning techniques to problems that are outside of the usual deep learning fields (vision, natural language).

Not all of the examples will work. Some of them are far to simple to even be considered viable trading strategies and are only presented for educational purposes. Others, in the notebook form I present, have not been trained for the proper amount of time. Perhaps with a bit of rented GPU time they will be more promising and I leave that as an excercise for the reader. Hopefully this project inspires some to try using deep learning techniques for some more interesting problems. [Contact me](mailto:ljrconnell@gmail.com) if interested in learning more or if you have suggestions for additions or improvements. 

The algorithms increase in complexity and introduce new concepts as they progress:

##Simple Regression [(notebook)][1]
Here we regress the prices from the last 100 days to the next day's price, training *W* and *b* in the equation *y = Wx + b* where *y* is the next day's price, *x* is a vector of dimension 100, *W* is a 100x1 matrix and *b* is a 1x1 matrix. We run the gradient descent algorithm to minimize the mean squared error of the predicted price and the actual next day price. Congratulations, you passed highschool stats. But hopefully this simple and naive example helps demonstrate the idea of a tensor graph, as well as showing a great example of extreme overfitting. 

##Simple Regression on Multiple Symbols [(notebook)][2] 
Things get a little more interesting as soon as we introduce more than one symbol. What is the best way to model our eventual investment strategy? We start to realize that our model only vaguely implies a policy (investment actions) by predicting the actual movement in price. The implied policy is simple: buy if the the predicted price movement is positive, sell if it is negative. But that doesnt sound realistic at all. How much do we buy? And will optimizing this, even if we are very careful to avoid overfitting, even produce results that allign with our goals? We havent actaully defined our goals explicitly, but for those who are not familiar with investment metrics, common goals include:
+ maximize risk adjusted return (like the [Sharpe](https://en.wikipedia.org/wiki/Sharpe_ratio) ratio)
+ consistency of returns over time
+ low market exposure
+ [long/short equity](http://www.investopedia.com/terms/l/long-shortequity.asp) 

If markets were easy to figure out and we could accurately predict the next day's return then it wouldn't matter. Our implied policy would fit with some goals (not long/short equity though) and the strategy would be viable. The reality is that our model cannot accurately predict this, nor will our strategy ever be perfect. Our best case scenario is always just winning slightly more than losing. When operating on these margins it is much more important that we consider the policy explicitly, thus moving to 'Policy Based' deep learning. 

##Policy Gradient Training [(notebook)][3]
Our policy will remain simple. We will chose a position, long/neutral/short, for each symbol in our portfolio. But now, instead of letting our estimation of the future return inform our decision, we train our network to choose the best position. Thus, instead of having an *implied* policy, it is *explicit* and *trained* directly. 
 Even thought the policy is simple in this case, training it is a bit more involved. I did my best to interpret Andrej Karpathy's excelent article on [Reinforcement Learning](http://karpathy.github.io/2016/05/31/rl/) when writing this code. It might be worth reading his explanation, but I'll do my best to summarize what I did.
 
We update our regression engine so that its output, *y*, is a vector in dimension *[batch_size, number_positions x number_symbols]* (a long, short, neutral bucket for each symbol). 

For each symbol in our portfolio, we **sample** the probability distribution of our three position buckets to get our policy decision (a position long/short/neutral), we multiply our decision (element of {-1,0,1}) by the target value to get a daily return for the symbol. Then we add that value for all the symbols to get a full daily return. We can also get other metrics like the total return and sharpe ratio since we actually are feeding this through as a batch (more on that later). As Karpathy points out, we are *only interested in the gradients of the positions we sampled*, so we select the appropriate columns from the output and combine them into a new tensor. 

This part of the code was a bit tricky for several reasons. First off, we have a loop through and isolate each symbol since I need one position per symbol. I also am using multinomial probability distributions, so I need to take a softmax of those values. Softmax pushes values so that they sum to 1, and therefore can represent a probability distribution. In pseudocode: `softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))`. 

```
for i in range(len(symbol_list)):
    symbol_probs = y[:,i*num_positions:(i+1)*num_positions]
    symbol_probs_softmax = tf.nn.softmax(symbol_probs)
```

Next, we sample that probability distribution. Even though the code is a nice one-liner due to tensorflow's multinomial function, the function is NOT DIFFERENTIABLE, meaning that we will not be able to "move through" this step durring back propogation. We calcultate the position vector simply by subtracting 1 from the column indices that we got from the sample so that we get and element of {-1,0,1}.

```
pos = {}
for i in range(len(symbol_list)):
    # ISOLATE from before
    # ... 
    sample = tf.multinomial(tf.log(symbol_probs_softmax), 1)
    pos[i] = tf.reshape(sample, [-1]) - 1   # choose(-1,0,1)
```

Then we multiply that position by the target (future return) for each day. This gives us our return. It already looks like a cost function but remember that it's not differentiable. 

```
symbol_returns = {}
for i in range(len(symbol_list)):
    # ISOLATE and SAMPLE from before
    # ...
    symbol_returns[i] = tf.mul(tf.cast(pos[i], float32),  y_[:,i])
```

Finally, we isolate the relevant column (the one we chose in our sample) from our probability distribution. The idea isn't very difficult but the code was a bit tough, remember that we are dealing with a whole batch of outputs at a time. This step NEEDS to be differentiable since we will use this tensor to compute our gradients. Unfortunately tensorflow is still developing a function that does it by itself, [here](https://github.com/tensorflow/tensorflow/issues/206) is the discussion. I actually think that my solution is the best and I suggest it in the discussion, but I'm really not an expert at efficient computation. If anyone thinks of a more efficient solution or if tensorflow finishes theirs please let me know. 

```
relevant_target_column = {}
for i in range(len(symbol_list)):
    # ...
    # ...
    sample_mask = tf.reshape(tf.one_hot(sample, 3), [-1,3])
    relevant_target_column[i] = tf.reduce_sum(symbol_probs_softmax * sample_mask,1)
```

So here is all of that together:

```
# loop through symbols, taking the buckets for one symbol at a time
pos = {}
symbol_returns = {}
relevant_target_column = {}
for i in range(len(symbol_list)):
    # ISOLATE the buckets relevant to the symbol and get a softmax as well
    symbol_probs = y[:,i*num_positions:(i+1)*num_positions]
    symbol_probs_softmax = tf.nn.softmax(symbol_probs)
    # SAMPLE probability to chose our policy's action
    sample = tf.multinomial(tf.log(symbol_probs_softmax), 1)
    pos[i] = tf.reshape(sample, [-1]) - 1   # choose(-1,0,1)
    # GET RETURNS by multiplying the policy (position taken) by the target return (y_) for that day
    symbol_returns[i] = tf.mul(tf.cast(pos[i], float32),  y_[:,i])
    # isolate the output probability the selected policy (for use in calculating gradient)
    sample_mask = tf.reshape(tf.one_hot(sample, 3), [-1,3])
    relevant_target_column[i] = tf.reduce_sum(symbol_probs_softmax * sample_mask,1)
```
 
So now we have a tensor with the regression's probability for the chosen (sampled) action for each symbol and each day. We also have a few performance metrics like daily and total return to choose from, but they're not differentiable because we sampled the probability so we cant just "gradient descent maximize" the profit...unfortunately. Instead, we find the sigmoid cross entropy (a sort of distance function) between the first table (the probabilities we chose/sampled) and an all-ones tensor of the same shape. We get a table of cross entropies of the same size (number of symbols by batch size) This is basically equivalent to saying, how do I do MORE of what I'm already doing, for every decision that I made. 
```
training_target_cols = tf.concat(1, [tf.reshape(t, [-1,1]) for t in relevant_target_column.values()])
ones = tf.ones_like(training_target_cols)
gradient = tf.nn.sigmoid_cross_entropy_with_logits(training_target_cols, ones)
```

Now we dont necessarilly want MORE of what we're doing, but the opposite of it is definitely LESS of it, which is useful. We multiply that tensor by our fitness function (the daily or aggregate return) and we use the gradient descent optimizer to minimize the cost. So you see? If the fitness function is negative, it will train the weights of the regression to NOT do what it just did. Its a pretty cool idea and it can be applied to a lot of problems that are much more interesting. I give some examples in the notebook about which different fitness functions you can apply which I think is better explained by seeing it. Here is that code:
```
cost = tf.mul(gradient , returns)        #returns are some reshaped version of symbol_returns (from above)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
```


##Stochastic Gradient Descent [(notebook)][4] 

As you saw in the notebook, the policy gradient doesnt train very well when we are grading it on the return over the entire dataset, but it trains very well when it uses each day's return or the position on each symbol every day. This makes sense, if we just take the total return over several years and its slightly positive then we tell our machine to do more of that. That will do almost nothing since so many of those decisions were actually losing money. The problem, as we have it set up now, needs to be broken down into smaller units. Fortunately there is some mathematical proof that this is legal and even faster. Score! 

Stochastic Gradient Descent is basically just breaking your data into smaller batches and doing gradient descent on each one. It will have slightly less accurate gradients WRT to the entire dataset's cost function, but since you are able to iterate faster with smaller batches you can run way more of them. There might even be more advantages to SGD that I'm not even mentioning, so you can read the [wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) or hear [Andrew Ng](https://www.coursera.org/learn/machine-learning#syllabus) talk about it. Or just use it since it works and its faster. If you're going on a wikipedia learning binge you might as well also learn about [Momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) and [Adagrad](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad) which are just variations. The latter two only really useful for people doing much bigger projects. If you are working on a huge project and your twobucksanhour AWS GPU instance is too slow then you should definitely be using them (and not be reading this introductory tutorial). At the higher level, the problem of tuning the optimizer and overall efficiency have been thoroughly [researched](http://www.andrewng.org/portfolio/on-optimization-methods-for-deep-learning/). 

##Multi Sampling [(notebook)][5]

Since we are sampling the policy, we can sample repeatedly in order to compute better. Karpathy's article summarizes the math behind this nicely and [this](http://arxiv.org/abs/1506.05254) paper is worth reading. The concept is intuitive and simple, but getting the math to work out and the tensor graph in order is very involved. One realizes that a mastery of numpy and a solid understanding of linear algebra are very important to tensorflow once the problems get...deeper, I guess is the word.

Multi sampling adds a useful computational kick that lets the network train much more efficiently. The results are already impressive. Using batches of less than 75 days and only training on the total return over that timeframe, we are able to "overfit" our network. Keep in mind that all we are doing is telling the network to do more of what it is doing when it does well, and less when it does poorly! Sure, we are still far away from having anything worthwhile out of sample, but that is because we are still using *linear regression*. 

By now you are probably either wondering *does this guy even know what deep learning is? I havent seen a single neural network!* or you completely forgot we were still using the same linear regression that 16 year olds learn in math class. Well, we'll get to neural networks next but I wanted to talk about other things before neural networks to show how much tensorflow can be used for before neural networks even get mentioned, and to show how much important math exists in deep learning that has nothing to do with neural networks. Tensorflow makes neural nets so easy that you barely even notice that they're part of the picture and its definitely not worth getting bogged down by their math if you dont have a solid understanding of the math behind cross entropy, policy gradients and the like. They probably even distract from where the true difficulty is. Maybe I'll try to get a regression to play pong so that everyone shuts about neural networks and starts talking about policy learning...

##Neural Networks [(notebook)][6]

So we finanlly get to it. Here's the same thing with a neural network. Its the simplest kind of net that there is but it is still very powerful. Way more powerful that our puny regression ever was becuase it has nonlinearities (RELU layers) between the other layers (which are basically just regressions by themselves). A sequence of linear regressions is still obviously linear<sup>[1](#myfootnote1)</sup>. But if we put a nonlinearity between the layers, then our net can do *anything*. Thats what gets people excited about neural networks, becuase they can hold enourmous amounts of information when trained well. Fortunatly, we just learned a bunch of cool ways to train them in steps 1-5. Now putting in the network is very easy. I really changed nothing except the Variable and the equation really is basically still *y = Wx + b* except non-linear *W*. 

The reason I introduced the networks so late is becuase they can be a bit difficult to tune. Chosing the right size of your network and the right training step can be difficult and sometimes it is helpful to start out simple until you have all the bells and whistles in place. 
 
In steps 3-5, we spent a lot of time figuring out tricks to do with the training step, which is a widely researched area at the moment and is probably more relevant to algorithmic trading that anything else. Now we are starting to demonstrate some of the techniques used in the prediction engine (regression before, neural network now). I believe this is a much more researched area and TensorFlow is better equiped for it. Many people describe the types of neural networks that we will learn as cells or *legos*. You dont need to think that much about how it works as long as you know what it does. If you noticed, thats what I did with the neural network. There is a lot more to learn and its worth learning, but when you're actually building with it, you dont think about RELU layers as much as input/output and a black box in the middle. Or at least *I* do...there are a bunch of people in image processing who look inside networks and do [very cool things](https://github.com/google/deepdream). 

##Regularization and Modularization [(notebook)][7]

From [Wikipedia][wiki_reg])), "regularization is the introduction of additional information in order to solve an ill-posed problem or to prevent overfitting." For our purposes, it is any technique used to reduce overfitting. One of the simplest yet most important examples is [early stopping][early_stopping], that is, ending gradient descent before there is no significant gain in evaluation performance. The idea is that once the model stops improving, it will start to overfit the training data. Most overfitting can be avoided with just this technique. 

We have more overfitting problems. Financial data is very noisey with faint signals that are very complex. Markets are zero sum and there are already an enormous number of very smart people getting paid a very large amount of money to develop trading strategies. There are many different patterns that can be learned but we can assume that all of the simple ones have been found (and therefore traded out of the market). Also, the market structure and properties change with time so even if we found a great pattern for the past five years, it might not work a month from now. 

We therefore need even more strategies to prevent overfitting. One strategy is called [L2 regularization](http://www.kdnuggets.com/2015/04/preventing-overfitting-neural-networks.html/2). Basically, we *punish a network for having very large weights* by adding the L2 norm of the weights, *1/2(||W||^2_2)*, times a constant *B* (to determine how much we want to regularize). It allows us to control the level of model complexity.

Another method is called [dropout regularization](http://arxiv.org/abs/1207.0580), a technique developed by Geoffrey Hinton. Overfitting is avoided by randomly ommitting half of the neurons during each training iteration. It sounds kooky but the idea is that it avoids the co-adaptation of features in the neural network, so that recognizing a certain set of features does not imply another set, as is the case in many overtrained nets. This way, each neuron will learn to detect a feature that is generally helpful. In validation and normal use, the neurons are not dropped so as to use all informaation. The technique makes the model more robust, avoids overfitting, and essentially turns your network into an ensemble that reaches consensus. Tensorflow's dropout layer includes the scaling required to safely ensemble when it comes time for validation. 

Since we want the same model with two different configurations--one with dropout active and one without--we will now pursue our long-overdue duty of refactoring our code. Although the code is presented in a single notebook, keep in mind that what I am essentially doing is modularizing. I won't claim that I am the most organized with my code, but I tried to keep it consistent with the best practices that I have observed others using with their open projects. Moving forward we will keep this structure becuase it is clearly superior to the organization that I had used before. 

##LSTM [(notebook)][8]

My favorite neural network, and a true stepping stone into real deep learning is the long short-term memory network, or LSTM. [Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) wrote an incredibly clear explanation of LSTM and there is really no substitute to reading his post. To describe the setup as briefly as possible, you input the data one timestep at a time to the LSTM cell. And each timestep the cell not only recieves the new input, but it recieves the last timestep's output and what is called the **cell state**, a vector that carries information about what happened in the past. Within the cell you have trained gates (basically small neural nets) that decide, based on the three inputs, what to forget from the past cell state, what to remember (or *add*) to the new state, and what to output this timestep. It is a very powerful tool and fascinating in how [effective it is](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). 

I am now pretty far into this series and I have a pretty good idea of where it will go from here. These are the problems that I must tackle in no particular order:
* new policies such as
    + long/short equality amongs two symbols and more
    + spread trading (if that is different from above)
    + minimize correlation/ new risk meaesure that is appropriate for large number of symbols
* migrating to AWS and using GPU computing power
* ensebling large numbers of strategies that are generated with the same code
    + policy grads find local maxima so no reason not to use that to my advantage
* testing suite to be able to test if the strategies are viable objectively
* convolution nets, especially among larger groups of symbols we can expect that some patterns are fractal
* turning it into a more formal project or web app

And of course we can start moving to other sources of data:
* text
* scraping
* games

Stay tuned for some articles that I will write about the algorithms used here and a discussion of the difficulties of using these techniques for algorithmic trading developement.  

<a name="myfootnote1">1</a>: I hope you didnt sleep through Linear Algebra class! I owe all my LA skills to [Comrade Otto Bretscher](http://personal.colby.edu/personal/o/obretsch/) of Colby College whose class I did sleep through but whose [text book](https://www.amazon.com/Linear-Algebra-Applications-Otto-Bretscher/dp/0136009263) is worth its weight in gold.

[early_stopping]: https://en.wikipedia.org/wiki/Regularization_(mathematics)#Early_stopping
[wiki_reg]: https://en.wikipedia.org/wiki/Regularization_(mathematics)
[1]: /notebooks/TF-FIN-1-singlestock_regresion.ipynb
[2]: /notebooks/TF-FIN-2-multistock_regresion.ipynb
[3]: /notebooks/TF-FIN-3-regression_with_policy_training.ipynb
[4]: /notebooks/TF-FIN-4-stochastic_gradient_descent.ipynb
[5]: /notebooks/TF-FIN-5-multi_sampling.ipynb
[6]: /notebooks/TF-FIN-6-neural_network.ipynb
[7]: /notebooks/TF-FIN-7-regularization_modular.ipynb
[8]: /notebooks/lstm_(7).ipynb#
