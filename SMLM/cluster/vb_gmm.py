import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


class VBGMM:
    """Fit a Bayesian GMM with for a fixed number of clusters with variable concentration parameter"""
    def __init__(self,X):
        self.X = X

    def plot_ellipses(self, ax, weights, means, covars):
        for n in range(means.shape[0]):
            eig_vals, eig_vecs = np.linalg.eigh(covars[n])
            unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
            angle = 180 * angle / np.pi
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
            ell = mpl.patches.Ellipse(
                means[n], eig_vals[0], eig_vals[1], angle=180 + angle, edgecolor="black"
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(weights[n])
            ell.set_facecolor("#56B4E9")
            ax.add_artist(ell)


    def plot_results(self, ax1, ax2, estimator, X, title, plot_title=False):
        ax1.set_title(title)
        ax1.scatter(X[:, 0], X[:, 1], s=2, marker="x", color='cornflowerblue', alpha=0.8)
        ax1.set_xticks(())
        ax1.set_yticks(())
        self.plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

        ax2.get_xaxis().set_tick_params(direction="out")
        ax2.yaxis.grid(True, alpha=0.7)
        for k, w in enumerate(estimator.weights_):
            ax2.bar(
                k,
                w,
                width=0.9,
                color="#56B4E9",
                zorder=3,
                align="center",
                edgecolor="black",
            )
        ax2.set_ylim(0.0, 1.1)
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax2.tick_params(axis="x", which="both", top=False)

        if plot_title:
            ax1.set_ylabel("Estimated Mixtures")
            ax2.set_ylabel("Weight of each component")

    def fit(self):
        random_state, n_components, n_features = 2, 12, 2
        estimators = [
            (
                "Finite mixture with a Dirichlet distribution\nprior and " r"$\gamma_0=$",
                BayesianGaussianMixture(
                    weight_concentration_prior_type="dirichlet_distribution",
                    n_components=2 * n_components,
                    reg_covar=0,
                    init_params="random",
                    max_iter=1500,
                    mean_precision_prior=0.8,
                    random_state=random_state,
                ),
                [0.001, 1, 1000],
            ),
            (
                "Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
                BayesianGaussianMixture(
                    weight_concentration_prior_type="dirichlet_process",
                    n_components=2 * n_components,
                    reg_covar=0,
                    init_params="random",
                    max_iter=1500,
                    mean_precision_prior=0.8,
                    random_state=random_state,
                ),
                [1, 1000, 100000],
            ),
        ]

        for title, estimator, concentrations_prior in estimators:
            plt.figure(figsize=(4.7 * 3, 8))
            plt.subplots_adjust(
                bottom=0.04, top=0.90, hspace=0.05, wspace=0.05, left=0.03, right=0.99
            )

            gs = gridspec.GridSpec(3, len(concentrations_prior))
            for k, concentration in enumerate(concentrations_prior):
                estimator.weight_concentration_prior = concentration
                estimator.fit(self.X)
                self.plot_results(
                    plt.subplot(gs[0:2, k]),
                    plt.subplot(gs[2, k]),
                    estimator,
                    self.X,
                    r"%s$%.1e$" % (title, concentration),
                    plot_title=k == 0,
                )
            plt.show()
