import numpy as np


def _get_im2col_indices(x_shape, kernel_h, kernel_w, padding=0, stride=1):
    n_samples, channels, height, width = x_shape
    out_h = (height + 2 * padding - kernel_h) // stride + 1
    out_w = (width + 2 * padding - kernel_w) // stride + 1

    if (height + 2 * padding - kernel_h) % stride != 0:
        raise ValueError("Invalid height/stride configuration for convolution.")
    if (width + 2 * padding - kernel_w) % stride != 0:
        raise ValueError("Invalid width/stride configuration for convolution.")

    i0 = np.repeat(np.arange(kernel_h), kernel_w)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_h), out_w)

    j0 = np.tile(np.arange(kernel_w), kernel_h)
    j0 = np.tile(j0, channels)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), kernel_h * kernel_w).reshape(-1, 1)
    return k, i, j


def im2col_indices(x, kernel_h, kernel_w, padding=0, stride=1):
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    k, i, j = _get_im2col_indices(x.shape, kernel_h, kernel_w, padding, stride)
    cols = x_padded[:, k, i, j]
    channels = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel_h * kernel_w * channels, -1)
    return cols


def col2im_indices(cols, x_shape, kernel_h, kernel_w, padding=0, stride=1):
    n_samples, channels, height, width = x_shape
    height_padded, width_padded = height + 2 * padding, width + 2 * padding
    x_padded = np.zeros((n_samples, channels, height_padded, width_padded), dtype=cols.dtype)

    k, i, j = _get_im2col_indices(x_shape, kernel_h, kernel_w, padding, stride)
    cols_reshaped = cols.reshape(channels * kernel_h * kernel_w, -1, n_samples)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_h * kernel_w
        self.weights = np.random.randn(out_channels, in_channels, kernel_h, kernel_w) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros((1, out_channels))

        self.inputs = None
        self.input_cols = None
        self.input_shape = None

        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs):
        self.inputs = inputs
        self.input_shape = inputs.shape

        n_samples, _, height, width = inputs.shape
        out_h = (height + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_w) // self.stride + 1

        self.input_cols = im2col_indices(
            inputs,
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            padding=self.padding,
            stride=self.stride,
        )

        weights_col = self.weights.reshape(self.out_channels, -1)
        output = weights_col @ self.input_cols + self.biases.reshape(-1, 1)
        output = output.reshape(self.out_channels, out_h, out_w, n_samples).transpose(3, 0, 1, 2)
        return output

    def backward(self, error):
        n_samples, _, out_h, out_w = error.shape
        error_reshaped = error.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.d_biases = np.sum(error, axis=(0, 2, 3), keepdims=False).reshape(1, -1)
        self.d_weights = (error_reshaped @ self.input_cols.T).reshape(self.weights.shape)

        weights_col = self.weights.reshape(self.out_channels, -1)
        d_input_cols = weights_col.T @ error_reshaped
        d_inputs = col2im_indices(
            d_input_cols,
            x_shape=self.input_shape,
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            padding=self.padding,
            stride=self.stride,
        )
        return d_inputs
