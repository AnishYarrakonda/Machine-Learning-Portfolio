import numpy as np

from cnn import col2im_indices, im2col_indices


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        if isinstance(pool_size, int):
            self.pool_h = self.pool_w = pool_size
        else:
            self.pool_h, self.pool_w = pool_size

        self.stride = stride
        self.inputs = None
        self.input_cols = None
        self.argmax = None

    def forward(self, inputs):
        self.inputs = inputs
        n_samples, channels, height, width = inputs.shape

        if (height - self.pool_h) % self.stride != 0 or (width - self.pool_w) % self.stride != 0:
            raise ValueError("Pooling window/stride does not align with input dimensions.")

        out_h = (height - self.pool_h) // self.stride + 1
        out_w = (width - self.pool_w) // self.stride + 1

        reshaped = inputs.reshape(n_samples * channels, 1, height, width)
        self.input_cols = im2col_indices(
            reshaped,
            kernel_h=self.pool_h,
            kernel_w=self.pool_w,
            padding=0,
            stride=self.stride,
        )
        self.argmax = np.argmax(self.input_cols, axis=0)
        pooled = self.input_cols[self.argmax, np.arange(self.argmax.size)]
        pooled = pooled.reshape(out_h, out_w, n_samples, channels).transpose(2, 3, 0, 1)
        return pooled

    def backward(self, error):
        n_samples, channels, out_h, out_w = error.shape
        height, width = self.inputs.shape[2], self.inputs.shape[3]

        d_input_cols = np.zeros_like(self.input_cols)
        error_flat = error.transpose(2, 3, 0, 1).reshape(-1)
        d_input_cols[self.argmax, np.arange(self.argmax.size)] = error_flat

        d_inputs = col2im_indices(
            d_input_cols,
            x_shape=(n_samples * channels, 1, height, width),
            kernel_h=self.pool_h,
            kernel_w=self.pool_w,
            padding=0,
            stride=self.stride,
        )
        d_inputs = d_inputs.reshape(self.inputs.shape)
        return d_inputs
