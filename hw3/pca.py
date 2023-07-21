from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    S = np.dot(np.transpose(dataset), dataset)
    return S / (len(dataset) - 1)


def get_eig(S, m):
    lam, u = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    lam = np.flip(lam, axis = 0)
    u = np.flip(u, axis = 1)
    lam = np.diag(lam)
    return lam, u


def get_eig_perc(S, perc):
    eigsum = 0
    for num in eigh(S, eigvals_only=True):
        eigsum += num
    lam, u = eigh(S, eigvals_only=False, subset_by_value=[eigsum*perc, np.inf])
    lam = np.flip(lam, axis=0)
    u = np.flip(u, axis=1)
    lam = np.diag(lam)
    return lam, u


def project_image(img, U):
    x = np.zeros((1024, 1))  # zero vector
    m = U.shape[1]  # shape of U
    image = img.reshape(1024, 1)  # reshape images to 1024
    for u in range(m):
        a = np.dot(np.transpose(U[:, [u]]), image)
        x = np.add(x, a * U[:, [u]])
    return np.transpose(x)[0]


def display_image(orig, proj):
    orig = np.transpose(orig.reshape(32, 32))  # reshape to 32*32 and rotate
    proj = np.transpose(proj.reshape(32, 32))

    figure, (axes1, axes2) = plt.subplots(ncols=2)
    axes1.set_title("Original")
    axes2.set_title("Projection")
    figure.colorbar(axes1.imshow(orig, aspect='equal'), ax=axes1)
    figure.colorbar(axes2.imshow(proj, aspect='equal'), ax=axes2)
    plt.show()