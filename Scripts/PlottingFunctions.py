import sys
sys.path.insert(0, '/AB_Testing')

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

from Scripts.ABtesting import ABTesting
ABT = ABTesting()

class PlottingFunctions:
    def _init_(self):
        """
        Initializing PlottingFunctions class
        """
        
    def plot_norm_dist(self, ax, mu, std, with_CI=False, sig_level=0.05, label=None):
        """
            Adds a normal distribution to the axes provided
            Example:
                plot_norm_dist(ax, 0, 1)  # plots a standard normal distribution
            Parameters:
                ax (matplotlib axes)
                mu (float): mean of the normal distribution
                std (float): standard deviation of the normal distribution
            Returns:
                None: the function adds a plot to the axes object provided
        """
        x = np.linspace(mu - 12 * std, mu + 12 * std, 1000)
        y = scs.norm(mu, std).pdf(x)
        ax.plot(x, y, label=label)

        if with_CI:
            self.plot_CI(ax, mu, std, sig_level=sig_level)
            
    def plot_CI(self, ax, mu, s, sig_level=0.05, color='grey'):
        """
        Calculates the two-tailed confidence interval and adds the plot to
        an axes object.
        Example:
            plot_CI(ax, mu=0, s=stderr, sig_level=0.05)
        Parameters:
            ax (matplotlib axes)
            mu (float): mean
            s (float): standard deviation
        Returns:
            None: the function adds a plot to the axes object provided
        """
        left, right = ABT.confidence_interval(sample_mean=mu, sample_std=s,
                                      sig_level=sig_level)
        ax.axvline(left, c=color, linestyle='--', alpha=0.5)
        ax.axvline(right, c=color, linestyle='--', alpha=0.5)
        
    def plot_null(self, ax, stderr):
        """
        Plots the null hypothesis distribution where if there is no real change,
        the distribution of the differences between the test and the control groups
        will be normally distributed.
        The confidence band is also plotted.
        Example:
            plot_null(ax, stderr)
        Parameters:
            ax (matplotlib axes)
            stderr (float): the pooled standard error of the control and test group
        Returns:
            None: the function adds a plot to the axes object provided
        """
        self.plot_norm_dist(ax, 0, stderr, label="Null")
        self.plot_CI(ax, mu=0, s=stderr, sig_level=0.05)
        
    def plot_alt(self, ax, stderr, mde):
        """
        Plots the alternative hypothesis distribution where if there is a real
        change, the distribution of the differences between the test and the
        control groups will be normally distributed and centered around d_hat
        The confidence band is also plotted.
        Example:
            plot_alt(ax, stderr, mde=0.025)
        Parameters:
            ax (matplotlib axes)
            stderr (float): the pooled standard error of the control and test group
        Returns:
            None: the function adds a plot to the axes object provided
        """
        self.plot_norm_dist(ax, mde, stderr, label="Alternative")
        
    def show_area(ax, mde, stderr, sig_level, area_type=None):
        """
        Fill between upper significance boundary and distribution for
        alternative hypothesis
        """
        left, right = ABT.confidence_interval(sample_mean=0, sample_std=stderr,
                                      sig_level=sig_level)
        x = np.linspace(-12 * stderr, 12 * stderr, 1000)
        null = ABT.ab_dist(stderr, 'control')
        alternative = ABT.ab_dist(stderr, mde, 'exposed')
        
        # if area_type is power
        # Fill between upper significance boundary and distribution for alternative
        # hypothesis
        if area_type == 'power':
            ax.fill_between(x, 0, alternative.pdf(x), color='green', alpha='0.25',
                            where=(x > right))
            ax.text(-5 * stderr, null.pdf(0),
                    'power = {0:.3f}'.format(1 - alternative.cdf(right)),
                    fontsize=12, ha='right', color='k')
            
         # if area_type is alpha
         # Fill between upper significance boundary and distribution for null
         # hypothesis
        if area_type == 'alpha':
             ax.fill_between(x, 0, null.pdf(x), color='blue', alpha='0.25',
                             where=(x > right))
             ax.text(-3 * stderr, null.pdf(0),
                     'alpha = {0:.3f}'.format(1 - null.cdf(right)),
                     fontsize=12, ha='right', color='k')

        # if area_type is beta
        # Fill between distribution for alternative hypothesis and upper
        # significance boundary
        if area_type == 'beta':
            ax.fill_between(x, 0, alternative.pdf(x), color='red', alpha='0.25',
                            where=(x < right))
            ax.text(-1 * stderr, null.pdf(0),
                    'beta = {0:.3f}'.format(alternative.cdf(right)),
                    fontsize=12, ha='right', color='k')
    def abplot(self, N_A, N_B, bcr, d_hat, sig_level=0.05, show_power=False,show_alpha=False, show_beta=False, show_p_value=False,show_legend=True):
        """Example plot of AB test
        Example:
            abplot(n=4000, bcr=0.11, d_hat=0.03)
        Parameters:
            n (int): total sample size for both control and test groups (N_A + N_B)
            bcr (float): base conversion rate; conversion rate of control
            d_hat: difference in conversion rate between the control and test
                groups, sometimes referred to as **minimal detectable effect** when
                calculating minimum sample size or **lift** when discussing
                positive improvement desired from launching a change.
        Returns:
            None: the function plots an AB test as two distributions for
            visualization purposes
        """
        # create a plot object
        fig, ax = plt.subplots(figsize=(12, 6))

        # define parameters to find pooled standard error
        X_A = bcr * N_A
        X_B = (bcr + d_hat) * N_B
        stderr = ABTesting.pooled_SE(N_A, N_B, X_A, X_B)

        # plot the distribution of the null and alternative hypothesis
        plot_null(ax, stderr)
        plot_alt(ax, stderr, d_hat)

        # set extent of plot area
        ax.set_xlim(-8 * stderr, 8 * stderr)

        # shade areas according to user input
        if show_power:
            show_area(ax, d_hat, stderr, sig_level, area_type='power')
        if show_alpha:
            show_area(ax, d_hat, stderr, sig_level, area_type='alpha')
        if show_beta:
            show_area(ax, d_hat, stderr, sig_level, area_type='beta')

        # show p_value based on the binomial distributions for the two groups
        if show_p_value:
            null = ab_dist(stderr, 'control')
            p_value = p_val(N_A, N_B, bcr, d_hat)
            ax.text(3 * stderr, null.pdf(0), 'p-value = {0:.3f}'.format(p_value), fontsize=12, ha='left')

        # option to show legend
        if show_legend:
            plt.legend()

        plt.xlabel('d')
        plt.ylabel('PDF')
        plt.show()     
            