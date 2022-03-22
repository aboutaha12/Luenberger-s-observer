# Luenberger observer
Our goal is to synthesize a luenberger observer for a damped pendulum with sinusoidal excitation using neural networks to learn the transformation z=T(x) and its inverse x=T^{-1}(z).

We will then test the robustness of the model by applying a sinusoidal excitation with a slightly variable frequency then with a larger frequency jump


**Organization of the different scripts**
## First step :
Define the different functions of the problem as well as the constants and the matrices that we have chosen for the observer.
The equations are :
$\begin{equation}\label{pendule}
\left\{
    \begin{array}{ll}
        \dot x_1 = x_2 \\
        \dot x_2 = -\textrm{sin}(x_1)-\mu x_2+u 
    \end{array}
\right.
, \quad y=x_1 \quad
\end{equation}$

You will find these functions in the script .

If you want to use the neural networks already trained go directly to step .

## Second step:
Create the datasets necessary for the training of the neural networks.

### What we do in this part :

In order to create the necessary data set for the training of the neural networks we consider a uniform distribution of each interval with 20 points in each dimension, we get a data set of $20^5=3,2.10^6$ $x$-points (with $x$ in $\mathbb{R}^5$).

We wish to obtain for each point $x$ the corresponding $z = T(x)$ in order to form couples $(x,z)$.
In this purpose, we integrate firstly backward the extended system during a time $t_c = 10$ (with $x$ as the inital condition) obtaining a point $x_{init-backwards}$.Then, we integrate forward both the system and the observer with an initial condition $x_{init-backwards}$ for the pendulum and $z_{init}=\begin{bmatrix}
    0,0,0,0,0,0 
\end{bmatrix}^{T}$ for the observer during the same time $t_c$ to obtain at the end a couple $(x,z)$.

The choose of $t_c$ is not arbitrary : 
We have:
$\begin{equation*}
    \overbrace{z-T(x)}^{\bullet} = A\,(z-T(x)) \Rightarrow \forall t \in [0,+\infty[,
z(t)=T(x(t))+[z_0-T(x_0)].\textrm{exp}(A.t)
\end{equation*}$

where $\textrm{exp}(A.t)$ is a diagonal matrix with the eigenvalues 
$$\{\textrm{exp}(-t),\textrm{exp}(-2t),\textrm{exp}(-3t),\textrm{exp}(-4t),\textrm{exp}(-5t),\textrm{exp}(-6t)\}$$

The smallest eigenvalue of $A_{pendulum}$ being -1 and considering $\textrm{exp}(-10)$ negligible we can say that after $t_c=10$ the observer converges to $T(x)$ and that the couple obtained after this procedure corresponds to $(x,T(x))$.

Applying this procedure to each of the $20^5$ points we obtain a data set of couples $(x,T(x))$ and we can train the neural networks on it.

We then splitted the data set into a training set and a test set (with a test size of $20\%$).Then, we prepossessed the data from the training set with a MinMaxScaler Transformation using the minimum and the maximum for each dimension in the data set to make all the components of the data set in $[0,1]$.

###  What you should execute:
In order to create the final datasets preprocessed run the script "2-Datasets_generation.py" .The last lines will save the train and test sets.
The lines 114-117 save the arrays min_x,max_x,min_z and max_z that represents the minimum and maximum of each component for the train set and are necessary for the preprocessing each time. For the dataset used here, you can already find these arrays in the repository.

## Third step:
Train of the neural network.

### What we do in this part:
Let $NN_1$ be the neural network for predicting $T(x)$ and $NN_2$ the neural network for predicting $x=T^{-1}(z)$.The basic architecture of each neural network is composed of one input layer with 5 neurons for $NN_1$ and 6 neurons for $NN_2$ (the dimension of the input), multi hidden layers with ReLu-based activation(3 hidden layers in our case) with 64 neurons each, and one output layer with a linear activation (since it is a regression problem) with 6 neurons for $NN_1$ and 2 neurons for $NN_2$. Indeed, since we only want to predict $(x_1,x_2)$ we can lighten the training of the neural network with an output of dimension 2 only (and not 5 as expected since we don't want to predict the input which is known). All the layers are dense layers. 

We also initialized the model with the Adam optimizer (an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data).

Given a loss function, the training of the neural network consists on adjusting its weights such as the loss between the predicted output and the real one is minimum.
Since we are facing a regression problem, we chose mean squared error as the loss function (the mean overseen data of the squared differences between true and predicted values).

The choice of the number of training epochs to use is also important for training neural networks. Using too many epochs can cause over-fitting of the training data set and also using too few epochs can cause under-fit model.We chose to train both neural networks with 50 epochs using an Early Stopping (a method that stops training once the model performance stops improving).

We can then train both $NN_1$ and $NN_2$ using the train set.

### What you should execute:
In order to train both neural networks execute the script "3-Train_Neural_Networks.py".You need to have gone through the second step to be able to obtain the correct datasets. However, if you have other datasets you can only replace the lines 8-11 by loading the datasets you want.

## Forth step:
Test the neural networks with the test set and with random points.

### What we do in this part:
\textit{Evaluation on the test set:}
Applying the neural network on the test set allows us to have an estimate of the performance of each neural network. In order to apply the model to each point of the test set we should firstly preprocess the data as we did for the training : we use the MinMaxScaler Transformation for each dimension with the minimum and maximum obtained on \textbf{the train set}.Then we apply $NN_1$ for each point $x_{\textrm{test-preprocessed}}$ and we compares the estimated $\hat{x}_{\textrm{test-preprocessed}}$ to $z_{\textrm{test-preprocessed}}$ (using the mean squared error).We do the same procedure for $NN_2$ (applying it to each $z_{\textrm{test-preprocessed}}$ and comparing the estimated $\hat{z}_{\textrm{test-preprocessed}}$ to the first two components of $x_{\textrm{test-preprocessed}}$ using the loss function).

The loss function (mean squared error) on the test set for $NN_1$ is $4,2.10^{-7}$ and for $NN_2$ is $10^{-6}$.

\textit{Evaluation on other random points:} In order to evaluate the performances of the models on new point that are not in the data set, we randomly choose 3000 points in $\mathcal{X}\times\mathcal{W}$.We apply the same method that we use on the synthesis of the data set to form for each point $x$ the couple $(x,T(x))$. After applying the MinMaxScaler Transformation for this couple, we predict $z_{predicted}$ using the $NN_1$ (with $x$ as the input of the neural network) and an $x_{predicted}$ using the $NN_2$ (with $z$ as the input).
Then, we can for each point have access to the mean squared error (for the couples ($x,x_{predicted}$) and ($z=T(x),z_{predicted}$)) and plot an histogram for each component with the logarithmic error in the $x$-axis and the number of points in the $y$-axis.

### What you should execute:

