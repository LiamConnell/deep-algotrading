#A tour through tensorflow with financial data

I present several models ranging in complexity from simple regression to LSTM and policy networks. The series can be used as an educational resource for tensorflow or deep learning, a reference aid, or a source of ideas on how to apply deep learning techniques to problems that are outside of the usual deep learning fields (vision, natural language).

Not all of the examples will work. Some of them are far to simple to even be considered viable trading strategies and are only presented for educational purposes. Others, in the notebook form I present, have not been trained for the proper amount of time. Perhaps with a bit of rented GPU time they will be more promising and I leave that as an excercise for the reader (who wants to make a lot of money). Hopefully this project inspires some to try using deep learning techniques for some more interesting problems. [Contact me](<ljrconnell@gmail.com>) if interested in learning more or if you have suggestions for additions or improvements. 

The algorithms increase in complexity and introduce new concepts in comments as they progress:
* [Simple Regression][1] regresses the prices from the last 100 days to the next days price, training *W* and *b* in the equation *y = Wx + b* where *y* is the next day's price, *x* is a vector of dimension 100, *W* is a 100x1 matrix and *b* is a 1x1 matrix. We run the gradient descent algorithm to minimize the mean squared error of the predicted price and the actual next day price. Congratulations you passed highschool stats. But hopefully this simple and naive example helps demonstrate the idea of a tensor graph, as well as showing a great example of extreme overfitting. 
* [Simple Regression on Multiple Symbols][2] things get a little more interesting as soon as we introduce more than one symbol. What is the best way to model our eventual investment strategy: our policy, if you will. We start to realize that our model only vaguely implies a policy (investment actions) by predicting the actual movement in price. The implied policy is simple: buy if the the predicted price movement is positive, sell if it is negative. But that doesnt sound realistic at all. How much do we buy? And will optimizing this, even if we are very careful to avoid overfitting, even produce results that allign with our goals? We havent actaully defined our goals explicitly, but for those who are not familiar with investment metrics, common goals include:
    + maximize risk adjusted return (like the [Sharpe](https://en.wikipedia.org/wiki/Sharpe_ratio) ratio)
    + consistency of returns over time
    + low market exposure
    + [long/short equity](http://www.investopedia.com/terms/l/long-shortequity.asp) 

 If markets were easy to figure out and we could accurately predict the next day's return then it wouldn't matter. Our implied policy would fit with some goals (not long/short equity though) and the strategy would be viable. The reality is that our model cannot accurately predict this, nor will our strategy ever be perfect. Our best case scenario is always winning slightly more than losing. When operating on these margins it is much more important that we consider the policy explicitly, thus moving to 'Policy Based' deep learning. 


Stay tuned for some articles that I will write about the algorithms used here and a discussion of the difficulties of using these techniques for algorithmic trading developement.  

[1]: /notebooks/singlestock_regresion_(1).ipynb
[2]: /notebooks/multistock_regresion_(2).ipynb
