### Activation Function Implementations:

Implementation of `activations.Linear`:

```python
class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        return dY

```

Implementation of `activations.Sigmoid`:

```python
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        return ...

```

Implementation of `activations.ReLU`:

```python
class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return np.maximum(0, Z)


    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        dZ = np.copy(Z)
        dZ[Z < 0] = 0
        dZ[Z >= 0] = 1
        return dY * dZ

```

Implementation of `activations.SoftMax`:

```python
class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        m = np.max(Z, axis =-1, keepdims=True)
        num = np.exp(Z - m)
        denom = np.sum(num, axis=-1, keepdims=True)
        return np.divide(num, denom)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        fwds = self.forward(Z)
        grads = []
        for idx, item in enumerate(fwds):
            diagonal = np.diag(item)
            item_reshaped = item.reshape(-1,1)
            diff = diagonal - item_reshaped.dot(item_reshaped.T)
            grad = np.dot(dY[idx], diff)
            grads.append(grad)

        return np.array(grads)

```


### Layer Implementations:

Implementation of `layers.FullyConnected`:

```python
class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []})  # type: ignore # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # type: ignore # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        W,b = self.parameters["W"], self.parameters["b"]
        Z = X.dot(W) + b
        # perform an affine transformation and activation
        out = self.activation(Z)
        
        # store information necessary for backprop in `self.cache`
        self.cache["X"] = X
        self.cache["Z"] = Z

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X, Z = self.cache["X"], self.cache["Z"]
        W = self.parameters["W"]
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdZ = self.activation.backward(Z, dLdY)

        dLdW = X.T.dot(dLdZ)
        dLdB = np.sum(dLdZ, axis=0, keepdims=True)

        dX = dLdZ.dot(W.T)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients["W"] = dLdW
        self.gradients["b"] = dLdB
        ### END YOUR CODE ###

        return dX

```

Implementation of `layers.Pool2D`:

```python
class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        kernel_height, kernel_width= self.kernel_shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        pad = self.pad

        out_rows = int((in_rows + 2*self.pad[0] - kernel_height) / self.stride + 1)
        out_cols = int((in_cols + 2*self.pad[1] - kernel_width) / self.stride + 1)
        # implement the forward pass
        if self.mode == "max":
            pool_fn = np.max
        elif self.mode == "average":
            pool_fn = np.mean
        else:
            raise ValueError("Invalid pooling mode")
        
        X_pad = np.pad(X, pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), mode='constant')

        # Initialize output volume
        X_pool = np.zeros((n_examples, out_rows, out_cols, in_channels))

        # Perform pooling
        for i in range(out_rows):
            for j in range(out_cols):
                i0 = i * self.stride
                i1 = i0 + kernel_height
                j0 = j * self.stride
                j1 = j0 + kernel_width
                # Apply pooling function over the window, preserving the channel dimension
                X_pool[:, i, j, :] = pool_fn(X_pad[:, i0:i1, j0:j1, :], axis=(1, 2))
        # cache any values required for backprop
        self.cache["X"] =X
        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        stride = self.stride
        X = self.cache["X"]
        
        out_rows = self.cache["out_rows"]
        out_cols = self.cache["out_cols"]

        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height, kernel_width = self.kernel_shape

        out_rows = int((in_rows + 2*self.pad[0] - kernel_height) / self.stride + 1)
        out_cols = int((in_cols + 2*self.pad[1] - kernel_width) / self.stride + 1)
        
        n_examples, in_rows, in_cols, in_channels = X.shape
        dX = np.zeros_like(X)
        # perform a backward pass
        for m in range(n_examples):
            for i in range(out_rows):
                for j in range(out_cols):
                    for c in range(in_channels):
                        # Calculate window boundaries, incorporating stride
                        i0, i1 = i * self.stride, (i * self.stride) + self.kernel_shape[0]
                        j0, j1 = j * self.stride, (j * self.stride) + self.kernel_shape[1]

                        if self.mode == "max":
                            # Extract the pooling region
                            pooling_region = X[m, i0:i1, j0:j1, c]
                            # Create a mask with True at the max value(s) in the pooling region, False elsewhere
                            mask = pooling_region == np.max(pooling_region)
                            # Distribute the gradient to the max location(s)
                            dX[m, i0:i1, j0:j1, c] += mask * dLdY[m, i, j, c]

                        elif self.mode == "average":
                            # Calculate the area of the pooling region
                            area = self.kernel_shape[0] * self.kernel_shape[1]
                            # Distribute the gradient uniformly across the pooling region
                            dX[m, i0:i1, j0:j1, c] += (dLdY[m, i, j, c] / area)
        
        if self.pad[0] > 0 or self.pad[1] > 0:
            dX = dX[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1], :]
        ### END YOUR CODE ###

        return dX

```

Implementation of `layers.Conv2D.__init__`:

```python
    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

```

Implementation of `layers.Conv2D._init_parameters`:

```python
    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

```

Implementation of `layers.Conv2D.forward`:

```python
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        ### BEGIN YOUR CODE ###
        out_rows = int((in_rows + 2 * self.pad[0] - kernel_height) / self.stride + 1)
        out_cols = int((in_cols + 2 * self.pad[1] - kernel_width) / self.stride + 1)
        
        # Pad the input
        X_pad = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')
        
        # Initialize the output volume
        Z = np.empty((n_examples, out_rows, out_cols, out_channels), dtype = X.dtype)
        # Apply the filters
        for i in range(out_rows):
            for j in range(out_cols):
                for k in range(out_channels):
                    # Determine the current slice
                    vertical_start = i * self.stride
                    vertical_end = vertical_start + kernel_height
                    horizontal_start = j * self.stride
                    horizontal_end = horizontal_start + kernel_width
                    
                    # Perform convolution
                    X_slice = X_pad[:, vertical_start:vertical_end, horizontal_start:horizontal_end, :]
                    weights = W[:, :, :, k]
                    conv = np.sum(X_slice * weights, axis=(1, 2, 3))
                    
                    # Add bias
                    Z[:, i, j, k] = conv + b[0, k]
        
     
        
        
        
        # implement a convolutional forward pass
        out = self.activation(Z)

        # cache any values required for backprop
        self.cache["X"] = X
        self.cache["Z"] = out
        ### END YOUR CODE ###

        return out

```

Implementation of `layers.Conv2D.backward`:

```python
    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
       
        ### BEGIN YOUR CODE ###
        W = self.parameters["W"]
        b = self.parameters["b"]
        X = self.cache["X"]
        Z = self.cache["Z"]
        stride, pad = self.stride, self.pad
        # perform a backward pass
        dZ = self.activation.backward(Z,dLdY)
        n_examples, in_rows, in_cols,in_channels = X.shape
        kernel_height, kernel_width, in_channels, out_channels = W.shape

        out_rows = int((in_rows + 2*self.pad[0] - kernel_height) / self.stride + 1)
        out_cols = int((in_cols + 2*self.pad[1] - kernel_width) / self.stride + 1)
        
        X_pad = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')
        dX_pad = np.zeros_like(X_pad)

        dW = np.zeros_like(W)
        dB = np.sum(dZ, axis=(0, 1, 2)).reshape(1, -1)

        for r in range(out_rows):
            for c in range(out_cols):
                for oc in range(out_channels):
                    dX_pad[:, 
                            r*self.stride:r*self.stride + kernel_height, 
                            c*self.stride:c*self.stride + kernel_width, 
                            :] += W[np.newaxis, :, :, :, oc] * dZ[:, r:r+1, c:c+1, np.newaxis, oc]
                
                    dW[:, :, :, oc] += np.sum(X_pad[:, 
                                            r*self.stride:r*self.stride + kernel_height, 
                                            c*self.stride:c*self.stride + kernel_width, :] * dZ[:, r:r+1, c:c+1, np.newaxis, oc], axis=0)
        dX = dX_pad[:, self.pad[0]:in_rows+self.pad[0], self.pad[1]:in_cols+self.pad[1], :] 
        
        self.gradients["W"] = dW
        self.gradients["b"] = dB
        ### END YOUR CODE ###
        
        return dX

```


### Loss Function Implementations:

Implementation of `losses.CrossEntropy`:

```python
class CrossEntropy(Loss):
    """Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        ### YOUR CODE HERE ###
        epsilon = np.finfo(float).tiny
        loss = -np.sum(Y * np.log(Y_hat +epsilon))/Y.shape[0]
        return loss

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the gradient of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        ### YOUR CODE HERE ###
        return -(1/Y.shape[0]) * (Y/Y_hat)

```


### Model Implementations:

Implementation of `models.NeuralNetwork.forward`:

```python
    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass through all the layers of the neural network.

        Parameters
        ----------
        X  design matrix whose must match the input shape required by the
           first layer

        Returns
        -------
        forward pass output, matches the shape of the output of the last layer
        """
        ### YOUR CODE HERE ###
        # Iterate through the network's layers.
        Y = X.copy()
        for layer in self.layers:
            Y=layer.forward(Y)
        return Y

```

Implementation of `models.NeuralNetwork.backward`:

```python
    def backward(self, target: np.ndarray, out: np.ndarray) -> float:
        """One backward pass through all the layers of the neural network.
        During this phase we calculate the gradients of the loss with respect to
        each of the parameters of the entire neural network. Most of the heavy
        lifting is done by the `backward` methods of the layers, so this method
        should be relatively simple. Also make sure to compute the loss in this
        method and NOT in `self.forward`.

        Note: Both input arrays have the same shape.

        Parameters
        ----------
        target  the targets we are trying to fit to (e.g., training labels)
        out     the predictions of the model on training data

        Returns
        -------
        the loss of the model given the training inputs and targets
        """
        ### YOUR CODE HERE ###
        # Compute the loss.
        # Backpropagate through the network's layers.
        loss_val = self.loss.forward(target, out)
        dLdY = self.loss.backward(target, out)
        for layer in reversed(self.layers):
            dLdY = layer.backward(dLdY)
        return loss_val

```

Implementation of `models.NeuralNetwork.predict`:

```python
    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make a forward and backward pass to calculate the predictions and
        loss of the neural network on the given data.

        Parameters
        ----------
        X  input features
        Y  targets (same length as `X`)

        Returns
        -------
        a tuple of the prediction and loss
        """
        ### YOUR CODE HERE ###
        # Do a forward pass. Maybe use a function you already wrote?
        # Get the loss. Remember that the `backward` function returns the loss.
        Y_hat = self.forward(X)
        loss = self.backward(Y, Y_hat)
        return Y_hat, loss

```

