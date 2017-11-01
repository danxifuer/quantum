# ref: https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
# good doc: https://github.com/thuijskens/bayesian-optimization
# good book: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
# paper: https://arxiv.org/abs/1610.08733

import numpy as np
import matplotlib.pylab as plt


def exponential_cov(x, y, params):
    return params[0] * np.exp(-0.5 * params[1] * np.subtract.outer(x, y) ** 2)


def conditional(x_new, x, y, params):
    A = exponential_cov(x_new, x_new, params)
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)

    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    return mu.squeeze(), sigma.squeeze()


theta = [1, 10]
sigma_0 = exponential_cov(0, 0, theta)
xpts = np.arange(-3, 3, step=0.01)
plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)

plt.show()
x = [1.]
y = [np.random.normal(scale=sigma_0)]
print(y)

sigma_1 = exponential_cov(x, x, theta)


def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    return y_pred, sigma_new


x_pred = np.linspace(-3, 3, 1000)
predictions = [predict(i, x, exponential_cov, theta, sigma_1, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")
plt.show()

m, s = conditional([-0.7], x, y, theta)
y2 = np.random.normal(m, s)
print(y2)

x.append(-0.7)
y.append(y2)

sigma_2 = exponential_cov(x, x, theta)
predictions = [predict(i, x, exponential_cov, theta, sigma_2, y) for i in x_pred]
y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")
plt.show()

x_more = [-2.1, -1.5, 0.3, 1.8, 2.5]
mu, s = conditional(x_more, x, y, theta)
y_more = np.random.multivariate_normal(mu, s)
print(y_more)

x += x_more
y += y_more.tolist()

sigma_new = exponential_cov(x, x, theta)
predictions = [predict(i, x, exponential_cov, theta, sigma_new, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")
plt.show()
