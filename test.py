import numpy as np

w, h = 3, 10
arr = np.random.rand(w, h)
xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

print(xs, '\n\n', ys)