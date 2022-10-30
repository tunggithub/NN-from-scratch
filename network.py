class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def complie_loss(self, loss):
        self.loss = loss
    
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                
                # compute loss (for display purpose only)
                err += self.loss.compute_loss(y_train[j], output)

                # backward propagation
                error = self.loss.derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
