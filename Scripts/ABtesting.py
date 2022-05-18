import scipy.stats as scs
class ABTesting:
    def _init_(self):
        """
        Initializing ABTesting class with functions used on ABTesting
        """
        
    def pooled_prob(self, Control, Exposed, X_A, X_B):
        """Returns pooled probability for two samples
        """
        return (X_A + X_B) / (Control + Exposed)
    
    def pooled_SE(self, Control, Exposed, X_A, X_B):
        """Returns the pooled standard error for two samples"""
        p_hat = self.pooled_prob(Control, Exposed, X_A, X_B)
        SE = np.sqrt(p_hat * (1 - p_hat) * (1/Control + 1/Exposed))
        return SE
    
    def z_val(self, sig_level=0.05, two_tailed=True):
        """
            Returns the z value for a given significance level
        """
        z_dist = scs.norm()
        if two_tailed:
            sig_level = sig_level/2
            area = 1 - sig_level
        else:
            area = 1 - sig_level
            
        z = z_dist.ppf(area)
        
        return z
    
    def confidence_interval(self, sample_mean=0, sample_std=1, sample_size=1,
                            sig_level=0.05):
        """
            Returns the confidence interval as a tuple
        """
        z = self.z_val(sig_level)
        
        left = sample_mean - z * sample_std/np.sqrt(sample_size)
        right = sample_mean + z * sample_std/np.sqrt(sample_size)
        
        return (left,right)
    
    def ab_dist(self, stderr, mde=0, group_type='control'):
        """
           Returns a distribution object depending on group type
           Examples:
           Parameters:
               stderr (float): pooled standard error of two independent samples
               mde (float): the mean difference between two independent samples
               group_type (string): 'control' and 'exposed' are supported
           Returns:
               dist (scipy.stats distribution object) 
        """
        if group_type == 'control':
            sample_mean = 0

        elif group_type == 'exposed':
            sample_mean = mde
            
        # create a normal distribution which is dependent on mean and std dev
        dist = scs.norm(sample_mean, stderr)
        
        return dist
    
    def p_val(self, Control, Exposed, p_A, p_B):
        """
            Returns the p_value for an A/B test
        """
        return scs.binom(Control, p_A).pmf(p_B * Exposed)
    
    
    # For classical testing
    def min_sample_size(self, bcr, mde, power=0.8, sig_level=0.05):
        """Returns the minimum sample size to set up a split test Arguments:
            bcr (float): probability of success for control, sometimes
            referred to as baseline conversion rate
            mde (float): minimum change in measurement between control
            group and test group if alternative hypothesis is true, sometimes
            referred to as minimum detectable effect
            power (float): probability of rejecting the null hypothesis when the
            null hypothesis is false, typically 0.8
            sig_level (float): significance level often denoted as alpha,
            typically 0.05
        Returns:
            min_N: minimum sample size (float)
        References:
            Stanford lecture on sample sizes
            http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
        """
        # standard normal distribution to determine z-values
        standard_norm = scs.norm(0, 1)

        # find Z_beta from desired power
        Z_beta = standard_norm.ppf(power)

        # find Z_alpha
        Z_alpha = standard_norm.ppf(1-sig_level/2)

        # average of probabilities from both groups
        pooled_prob = (bcr + bcr+mde) / 2

        min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2/ mde**2)

        return min_N