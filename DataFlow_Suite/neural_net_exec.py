#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas.io.data as web
import numpy as np
import os
import pandas as pd

# we modify this data organizing slightly to get two symbols
def get_prices(symbol):
    start, end = '2007-05-02', '2016-04-11'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data=pd.DataFrame(data)
    prices=data['Adj Close']
    prices=prices.astype(float)
    return prices

def get_returns(prices):
        return ((prices-prices.shift(-1))/prices)[:-1]
    
def get_data(list):
    l = []
    for symbol in list:
        rets = get_returns(get_prices(symbol))
        l.append(rets)
    return np.array(l).T

def sort_data(rets):
    ins = []
    outs = []
    for i in range(len(rets)-100):
        ins.append(rets[i:i+100].tolist())
        outs.append(rets[i+100])
    return np.array(ins), np.array(outs)
        
print('loading data')
symbol_list = ['C', 'GS']
def data_loader_1(symbol_list):
    if os.path.exists('rets.pkl'):
        rets = pd.read_pickle('rets.pkl')
    else:
        rets = get_data(symbol_list)
        pd.to_pickle(rets, 'rets.pkl')
    ins, outs = sort_data(rets)
    ins = ins.transpose([0,2,1]).reshape([-1, len(symbol_list) * 100])
    div = int(.8 * ins.shape[0])
    train_ins, train_outs = ins[:div], outs[:div]
    test_ins, test_outs = ins[div:], outs[div:]

    #normalize inputs
    train_ins, test_ins = train_ins/np.std(ins), test_ins/np.std(ins)
    return train_ins, test_ins, train_outs, test_outs
train_ins, test_ins, train_outs, test_outs = data_loader_1(symbol_list)
print('loaded data')


sess = tf.Session()

positions = tf.constant([-1,0,1]) #long, neutral or short
num_positions = 3
num_symbols = len(symbol_list)
num_samples = 10

n_input = num_symbols * 100
n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_classes = num_positions * num_symbols # MNIST total classes (0-9 digits)


# define placeholders 
x = tf.placeholder(tf.float32, [None, num_symbols * 100])
y_ = tf.placeholder(tf.float32, [None,  num_symbols])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
y = multilayer_perceptron(x, weights, biases)



# loop through symbol, taking the columns for each symbol's bucket together
pos = {}
sample_n = {}
sample_mask = {}
symbol_returns = {}
relevant_target_column = {}
for i in range(num_symbols):
    # isolate the buckets relevant to the symbol and get a softmax as well
    symbol_probs = y[:,i*num_positions:(i+1)*num_positions]
    symbol_probs_softmax = tf.nn.softmax(symbol_probs) # softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
    # sample probability to chose our policy's action
    sample = tf.multinomial(tf.log(symbol_probs_softmax), num_samples)
    for sample_iter in range(num_samples):
        sample_n[i*num_samples + sample_iter] = sample[:,sample_iter]
        pos[i*num_samples + sample_iter] = tf.reshape(sample_n[i*num_samples + sample_iter], [-1]) - 1
        symbol_returns[i*num_samples + sample_iter] = tf.mul(
                                                            tf.cast(pos[i*num_samples + sample_iter], np.float32), 
                                                             y_[:,i])
        
        sample_mask[i*num_samples + sample_iter] = tf.cast(tf.reshape(tf.one_hot(sample_n[i*num_samples + sample_iter], 3), [-1,3]), np.float32)
        relevant_target_column[i*num_samples + sample_iter] = tf.reduce_sum(
                                                    symbol_probs * sample_mask[i*num_samples + sample_iter],1)
    


daily_returns_by_symbol_ = tf.concat(1, [tf.reshape(t, [-1,1]) for t in symbol_returns.values()])
daily_returns_by_symbol = tf.transpose(tf.reshape(daily_returns_by_symbol_, [-1,2,num_samples]), [0,2,1]) #[?,5,2]
daily_returns = tf.reduce_mean(daily_returns_by_symbol, 2) # [?,5]

total_return = tf.reduce_prod(daily_returns+1, 0)
z = tf.ones_like(total_return) * -1
total_return = tf.add(total_return, z)


ann_vol = tf.mul(
    tf.sqrt(tf.reduce_mean(tf.pow((daily_returns - tf.reduce_mean(daily_returns, 0)),2),0)) ,
    np.sqrt(252)
    )
sharpe = tf.div(total_return, ann_vol)
#Maybe metric slicing later
#segment_ids = tf.ones_like(daily_returns[:,0])
#partial_prod = tf.segment_prod(daily_returns+1, segment_ids)


training_target_cols = tf.concat(1, [tf.reshape(t, [-1,1]) for t in relevant_target_column.values()])
ones = tf.ones_like(training_target_cols)
gradient_ = tf.nn.sigmoid_cross_entropy_with_logits(training_target_cols, ones)

gradient = tf.transpose(tf.reshape(gradient_, [-1,2,num_samples]), [0,2,1]) #[?,5,2]

#cost = tf.mul(gradient , daily_returns_by_symbol_reshaped)
#cost = tf.mul(gradient , tf.expand_dims(daily_returns, -1))
cost = tf.mul(gradient , tf.expand_dims(total_return, -1))
#cost = tf.mul(gradient , tf.expand_dims(sharpe, -1))

optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
costfn = tf.reduce_mean(cost)


def run_training_and_return_test_results():
    print('START TRAINING_______________________________')
    # initialize variables to random values
    init = tf.initialize_all_variables()
    sess.run(init)
    # run optimizer on entire training data set many times
    train_size = train_ins.shape[0]
    for epoch in range(20000):
        start = rng.randint(train_size-15)
        batch_size = rng.randint(2,50)
        end = min(train_size, start+batch_size)

        sess.run(optimizer, feed_dict={x: train_ins[start:end], y_: train_outs[start:end]})#.reshape(1,-1).T})
        # every 1000 iterations record progress
        if (epoch+1)%1000== 0:
            t,s, c = sess.run([ total_return, sharpe, costfn], feed_dict={x: train_ins, y_: train_outs})#.reshape(1,-1).T})
            t = np.mean(t)
            s = np.mean(s)
            print("Epoch:", '%04d' % (epoch+1), "cost=",c, "total return=", "{:.9f}".format(t), 
                 "sharpe=", "{:.9f}".format(s))
            #print(t)

    print('DONE TRAINING _______________________________')

    d_tr, t_tr = sess.run([daily_returns, total_return], feed_dict={x: train_ins, y_: train_outs})

    d_te, t_te = sess.run([daily_returns, total_return], feed_dict={x: test_ins, y_: test_outs})

    t_tr = np.mean(t_tr)
    t_te = np.mean(t_te)

    print("total return train=", "{:.9f}".format(t_tr))
    print("total return test=", "{:.9f}".format(t_te))
    return t_tr, t_te

results = []
pos_results = []
for i in range(100):
    tr, te = run_training_and_return_test_results()
    results.append(te)
    if tr>0:
        pos_results.append(te)
    print('iter: ', i)
    print(np.mean(results))
    print(np.mean(pos_results))
    #print('final from net', '%f' %zz)
    
print(np.mean(results))
print(np.mean(pos_results))
