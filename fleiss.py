import numpy as np
import statsmodels.stats.inter_rater as irr
# from statsmodels.stats.inter_rater import aggregate_raters

# Create a list of ratings for each rater
rater1 = [3, 3, 5, 3, 6, 2, 1, 4, 1, 2]
rater2 = [6, 5, 6, 2, 5, 1, 4, 5, 1, 1]
rater3 = [6, 3, 4, 1, 1, 6, 6, 2, 3, 2]

# Aggregate the ratings using Fleiss' kappa coefficient
agg = irr.aggregate_raters([rater1, rater2, rater3])

giro = np.array([rater1, rater2, rater3]).transpose()

res = irr.fleiss_kappa(irr.aggregate_raters(giro)[0], method='fleiss')
print(res)
# print("Fleiss' kappa coefficient: {:.2f}".format(kappa))
# print("p-value: {:.2f}".format(pvalue))
