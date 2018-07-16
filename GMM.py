import numpy as np
import matplotlib.pyplot as plt


def compute_gaussian(x, mu, sigma):
    x = np.reshape(x, newshape=(x.shape[0], 1))
    mu = np.reshape(mu, newshape=(mu.shape[0], 1))
    a = np.linalg.det(sigma)**(1/2) * (2*np.pi)**(x.shape[0]/2)
    b = np.exp(
        -1/2 * np.dot(
            np.dot(
                (x - mu).T,
                np.linalg.inv(sigma)
            ),
            x-mu
        )
    )
    return 1/a * b


def gmm_em(k=4):
    """Read data and plot figure"""
    data_obj = open("data.txt", 'r')

    data_list = eval(data_obj.readline())

    x1 = list()
    x2 = list()
    for data_simple in data_list:
        x1.append(data_simple[0])
        x2.append(data_simple[1])

    plt.figure('Data')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x1, x2, '.')
    # plt.show()

    """Init the parameters: alpha, mu and sigma"""
    # data_array.shape = [n, m]
    data_array = np.array([x1, x2])
    m = data_array.shape[1]
    n = data_array.shape[0]
    alpha = list()
    mu = list()
    sigma = list()
    for i in range(k):
        alpha.append(1/k)
        mu.append([data_array[0][i], data_array[1][i]])
        sigma.append(np.cov(data_array))

    # alpha.shape = [k, 1]
    alpha = np.expand_dims(np.array(alpha), -1)
    # mu.shape = [k, n]
    mu = np.array(mu)
    # sigma.shape = [k, n, n]
    sigma = np.array(sigma)

    """EM algorithm"""
    fig = plt.figure("EM algorithm")
    iteration = 0
    # gama.shape = [m, k]
    gama = np.zeros(shape=(m, k))
    while iteration != 60:
        iteration += 1
        # print(iteration)

        # E step
        for j in range(m):
            px = 0
            for l in range(k):
                px = px + alpha[l] * compute_gaussian(data_array.T[j], mu[l], sigma[l])
            for i in range(k):
                gama[j][i] = alpha[i]*compute_gaussian(data_array.T[j], mu[i], sigma[i])/px

        # M step
        alpha = np.sum(gama, axis=0)/m
        mu = np.dot(gama.T, data_array.T)/np.sum(gama.T, axis=-1, keepdims=True)

        x_minus_mu = np.expand_dims(data_array, 0) - np.expand_dims(mu, -1)
        for i in range(k):
            sigma[i] = np.dot(
                (np.expand_dims(gama.T, 1) * x_minus_mu)[i, :, :],
                np.transpose(x_minus_mu, (0, 2, 1))[i, :, :]
            )/np.sum(gama.T, axis=-1)[i]

        # report
        if iteration % 10 == 0:
            label = np.argmax(gama, axis=-1)
            subplot = fig.add_subplot(2, 3, iteration/10)
            subplot.set_title("After " + str(iteration) + " iterations")
            for i in range(m):
                if label[i] == 0:
                    subplot.scatter(data_array[0][i], data_array[1][i], color="red", marker='.')
                if label[i] == 1:
                    subplot.scatter(data_array[0][i], data_array[1][i], color="green", marker='.')
                if label[i] == 2:
                    subplot.scatter(data_array[0][i], data_array[1][i], color="yellow", marker='.')
                if label[i] == 3:
                    subplot.scatter(data_array[0][i], data_array[1][i], color="black", marker='.')
                if label[i] == 4:
                    subplot.scatter(data_array[0][i], data_array[1][i], color="blue", marker='.')
                if label[i] == 5:
                    subplot.scatter(data_array[0][i], data_array[1][i], color="pink", marker='.')

    plt.show()


if __name__ == "__main__":
    gmm_em()