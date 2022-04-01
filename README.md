# MNIST-Tensorflow-Explained
**MNIST Handwritten Digit Recognition in Keras** </br>
In this example we'll build a neural network and be able to recognize handwritten digits using the  [MNIST dataset](http://yann.lecun.com/exdb/mnist/) 
.


**Buikding the network** </br>
<p align='center'>
<img width=""300" height="200" src="https://cdn.nextjournal.com/data/1220CC01595BBCB08CCAC75AC0A373519699CFBC6FB7E6118A92DDB89EDB63490CFE?filename=text4298.png&content-type=image/png"> </br>
<em>Our densely-connected network with two hidden layers.   </em>
</p>
Our pixel vector serves as the input. Then, two hidden 512-node layers, with enough model complexity for recognizing digits. For the multi-class classification we add another densely-connected (or fully-connected) layer for the 10 different output classes. For this network architecture we can use the Keras Sequential Model. We can stack layers using the .add() method.

When adding the first layer in the Sequential Model we need to specify the input shape so Keras can create the appropriate matrices. For all remaining layers the shape is inferred automatically. 

In order to introduce nonlinearities into the network and elevate it beyond the capabilities of a simple perceptron we also add activation functions to the hidden layers. The differentiation for the training via backpropagation is happening behind the scenes without having to implement the details. 

We also add dropout as a way to prevent overfitting. Here we randomly keep some network weights fixed when we would normally update them so that the network doesn't rely too much on very few nodes.

The last layer consists of connections for our 10 classes and the softmax activation which is standard for multi-class targets.
```
# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))
```
