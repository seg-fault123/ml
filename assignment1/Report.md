<!-- ![Gradient Descent](./images/gd_iterations.png) -->

# Report for FML Assignment 1
#### Details of the submission:
* `models.py` file contains all the source code used to implement the solutions of the questions.

* `demonstration.py` file uses the models implemented in `models.py` to demonstrate and arrive at the solution. To run `demonstration.py`, keep the `models.py` in the same directory and change the `train_path` and `test_path` variables defined in the `demonstration.py` file to the corresponding path of the training and testing data.

* Libraries used:
    * `numpy` : for matrix and numerical operations
    * `pandas` : for loading and storing data in matrix format.
    * `matplotlib` : for plotting the graphs
    * `seaborn` : for plotting the graphs

* The measure of error used is the Mean Squared Error explained later.

## Deatils of the Source Code in `models.py`
* `mse(y_true, y_pred)`. This function calculates the Mean Squared Error of the predictions. 
    * `y_true` parameter is the true output variable.
    * `y_pred` parameter is the predicted output varaible.
    * Both parameters are expected to be `numpy arrays`.
    * The formula is $\frac{1}{n} \displaystyle \sum_{i=1}^n (y_i - \hat{y}_i)^2$. Where $y_i$ and $\hat{y}_i$ represents the true output variable and predicted output variable for the $i^{th}$ data point respectively. 

* `OLS`: This class implements the analytical solution of Linear Regression.
    * The hypothesis of this method is given by $y= \displaystyle \sum_{j=1}^d w_jx_j + w_0$. Where $w_j$ is the weight associated to the $j^{th}$ feature and $w_0$ is the inrecept value.
    * `fit(X, y)` method calculates the weights as given in the above formula. `X` is the data matrix that has data points as rows and features as columns, expected to be a `numpy ndarray` or a `pandas dataframe`. `y` is the output variable which is expected to be a `numpy array` or  a `pandas series`. `X` does not need to have the intercept column, the method adds the intercept column internally.
    * The weights are calculated and stored in the `coef_` variable of the class. The formula to calculate the weights is given by: $w_{ML} = (X^T X)^{-1} X^T y$. 
    * `predict(X)` method calcultes the predictions for data stored in `X`. `X` is the data matrix that has data points as rows and features as columns, expected to be a `numpy ndarray` or a `pandas dataframe`.
    * The predictions are calculated by the formula: $Xw_{ML}$

* `GradientDescent`: This class implements the gradient descent solution to Linear Regression.
    * The class expects `step_size` and `iterations` from the user that correspond to the gradient step size and total iterations to be performed. Similar to `OLS` class, the weights are stored in the `coef_` variable of the class.
    * `gradient(X, y)` method calcultes the gradient based on the the data and current weights. The requirements for `X` and `y` are the same as in `OLS`. The formula of gradient is given by : $\nabla f(w)=X^T Xw - X^T y$
    * The update rule is given by : $w_{t+1}=w_t - \eta \nabla f(w_t)$. Where $\eta$ is the step size.
    * `fit(X, y, w_ml)` implements the gradient descent algorithm as given by the above update rule. The requirements for `X` and `y` are the same as in `OLS`. Along with this, this method also finds the error between $w_{ML}$ and $w_t$ for each itertaion, where $w_{ML}$ represents the solution given by `OLS` method. The errors are calculted as: $||w_{ML}-w_t||_2$ and stored in `errors` variable of the class.
    * `predict(X)` method calcultes the predictions similar to the `OLS` class. The predictions are calculted as : $X\hat{w}$, where $\hat{w}$ is the weight vector obtained after running the gradient descent algorithm.

* `StochasticGradientDescent`: This class implements the stochastic gradient descent method for Linear Regression. The batch size is taken as 100 data points at a time. The class design is same as `GradientDescent` class. The only difference lies in the `fit(X, y, w_ml)` method.
    * `fit(X, y, w_ml)` implements the method and calcultes the weights. Instead of taking the whole dataset for the gradient update rule, it takes a random batch of 100 data points to calculate the gradient. The program is such that it ensures that each data point is selected in the training batch atleast once. Similar to `GradientDescent` class, this method also stores the error between $w_{ML}$ and $w_t$ for each iteration in the `errors` variable.

* `RidgeGD`: This class implements the Ridge Gradient Descent Algorithm. The class design is very similar to `GradientDescent` class, some key differences are highlighted below:
    * In addition to `step_size` and `iterations` this class also expects the `regu` parameter which corresponds to the regularization parameter in Ridge Regression.
    * `gradient(X, y)` method calcultes the gradient of the objective function of the Ridge formulation. Hence the formula is different as compared to `GradientDescent` and given by: $\nabla f(w)= X^T X w - X^T y + \lambda w$. The update rule is exactly same as `GradientDescent`.

* `KernelRegression`: This class implements the Polynomial Kernel Regression algorithm.
    * The class expects a `degree` parameter that specifies the degree of the kernel function. The class also stores the data passed during the `fit(X, y)` method in the `data` variable, as it will be required for calculating the Kernel Matrix. `X` and `y` have the same requirements as in `OLS`.
    * The formula of calculting the kernel function for two vectors is given by: $k(x_1, x_2)=(x_1^T x_2 + 1)^p$ where $p$ is the degree specified by the user. This value is same as taking dot product in higher dimension hence, $k(x_1, x_2)=\phi(x_1)^T \phi(x_2)$ where $\phi(x)$ represents the function that transforms $x$ to higher dimension.
    * Instead of calculating the optimal weights in the higher dimension, we calculate coefficients of combinations (given by $\alpha$) of data points that lead us to the optimal weights. So $\hat{w}= \phi(X)^T \alpha$
    * The formula for calculating $\alpha$ is given by: $\alpha=K^{-1}y$
    * `calc_kernel(X_new)` method calcultes the kernel matrix corresponding to `X_new` which represents a data matrix similar to `X` that was passed to `fit(X, y)` method. The formula for calculating the kernel matrix is given by: $K^{new}_{ij}= (<X^{new}_i, X_j> + 1)^p = k(X^{new}_i, X_j)$, where $<X^{new}_i, X_j>$ is the dot product between the $i^{th}$ data point (row) of $X^{new}$, and $j^{th}$ data point (row) of $X$ (data matrix on which the model has been trained). In numpy the whole matrix is calculted easily by `(X_new @ X.T + 1)**p`.
    * The prediction is made using the $\alpha$ and $K^{new}$ with the formula: $K^{new} \alpha$
    * `fit(X, y)` method calculates the $\alpha$
    * `predict(X)` method calcultes the predictions by calculating $K^{new}$ and using the already calculated $\alpha$.