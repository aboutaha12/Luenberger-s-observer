# KKL observer for a nonlinear system with inputs

Our goal is to synthesize a numerical nonlinear Luenberger observer (KKL observer) for a damped pendulum with measured position and sinusoidal excitation. Because the input is not known in advance, we design the observer for the class of sinusoidal inputs with constant frequency generated by a three-dimensional exosystem. In other words, we design a functional KKL observer for an autonomous extended system with input model reconstructing the state from the knowledge of the input and output.  Neural networks are used to learn the transformation into KKL contration coordinates and its inverse. 

We then test the robustness of this observer by applying to the system inputs that are not exactly in the considered class, namely sinusoidal excitations with a slightly variable frequency and then a larger frequency discontinuity.

This work was carried out by Taha Balki at the Centre Automatique et Systèmes, Mines Paris, Université PSL, during a research trimester supervised by Pauline Bernard and Florent Di Meglio. 

More details on the theory and design of functional KKL observers appear in 

*On the existence of robust functional KKL observers*, Mario Spirito, Pauline Bernard, Lorenzo Marconi, American Control Conference, 2022.

## Organization of the different scripts## 
## First step :
Define the different functions of the problem as well as the constants and the matrices that we have chosen for the observer.
The equations are :
```math
$$
\begin{equation}\label{pendule}
\left\{
    \begin{array}{ll}
        \dot x_1 = x_2 \\
        \dot x_2 = -\textrm{sin}(x_1)-\mu x_2+u 
    \end{array}
\right.
, \quad y=x_1 \quad
\end{equation}
$$
```
You will find these functions in the script '1-Constants_functions.py '.

If you want to use the neural networks already trained and only test the observer go directly to step 5.

## Second step:
Create the datasets necessary for the training of the neural networks.

### What we do in this part :

In order to create the necessary data set for the training of the neural networks we consider a uniform distribution of each interval with 20 points in each dimension, we get a data set of 20^5=3,2.10^6 x-points (with $x$ in $\mathbb{R}^5$).

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
You should execute the scirpt "4-Test_Neural_Networks.py". The lines 38-41 loads the arrays min and max (you can use the one in the repository). The lines 43-44 loads the test test obtained in step 2 and the lines 46-47 loads our model. You can change the model by correctly changing the lines above.

## Fifth step :
Application for predicting a real trajectory.

### What we do in this part:
Now that the neural networks are trained we will apply the $NN_2$ to the output $z$ of the system \eqref{obs_pendule} in order to predict $(x_1,x_2)$:

We initialize the system in  $(x_1_0 = 0.33019075, x_2_0 = -0.71512765)$ and integrate the principle equation during an arbitrary time $t_{simulation} = 40s$ (long enough to ensure the convergence of the observer) in order to have for all $t$ in $[0,t_{simulation}]$ the measure $y=(x_1(t),u(t))$ necessary for the observer,and the couple ($x_{real}_1(t),x_{real}_2(t)$) in order to compare it to the predicted one (in reality $y$ is directly measured with physical sensors but since we don't have access to real measures we use this procedure instead). Then, we use the observer during the same time $t_{simulation}$ with an arbitray initial condition (for instance $(0,0,0,0,0,0)$). Finally, we apply the $NN_2$ to each point $z(t)$ at a time $t\in[0,t_{simulation}]$ to obtain a couple ($x_{predicted}_1(t),x_{predicted}_2(t)$) that can be compared to ($x_{real}_1(t),x_{real}_2(t)$) to evaluate the performances of the observer along a trajectory.

* We apply in a first step this method with an input represented by a sinusuid Amp*np.cos(w*t-fi) of constant frequency w= 0.3189678255430453 and where 
$$
Amp = 0.5557902677321038 
fi = 0.3538347020102439 $$
We obtain the figures for z,T(x_real),T(x_predicted) and for x_real and x_predicted in the folder Fixed_freq.

* The, we apply the method to the same input Amp*np.cos(w*t-fi) but this time with a lightly variable frequency (which equals w/2 for t<16, w for t>24 and a polynomial interpolation of order 3 between the two).
We obtain the figures for z,T(x_real),T(x_predicted) and for x_real and x_predicted in the folder Lightly_var.


* Last we apply the method to the same input Amp*np.cos(w*t-fi) but this time with a sudden jump in frequency (which equals w/2 for t<16, w for t>18 and a polynomial interpolation of order 3 between the two).
We obtain the figures for z,T(x_real),T(x_predicted) and for x_real and x_predicted in the folder Jump_freq.

### What you need the execute:
In order to obtain all these figures, you only need to execute the script "Trajectory" using the arrays min_x,max_x,min_z,max_z and the models in the repository (but you can modify it respectively in the lines 47-50 and 53-54) 
