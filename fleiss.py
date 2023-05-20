import numpy as np
import statsmodels.stats.inter_rater as irr
# from statsmodels.stats.inter_rater import aggregate_raters

# Create a list of ratings for each rater
rater1 = [2, 3, 5, 1, 6, 2, 1, 3, 1, 2] 
rater2 = [2, 3, 4, 1, 5, 1, 4, 5, 1, 3]
rater3 = [2, 3, 4, 1, 1, 2, 3, 4, 1, 2]

# Aggregate the ratings using Fleiss' kappa coefficient
agg = irr.aggregate_raters([rater1, rater2, rater3])

giro = np.array([rater1, rater2, rater3]).transpose()

res = irr.fleiss_kappa(irr.aggregate_raters(giro)[0], method='fleiss')
print(res)
# print("Fleiss' kappa coefficient: {:.2f}".format(kappa))
# print("p-value: {:.2f}".format(pvalue))

# Fleiss' kappa can range from -1 to +1.

# test for generated or not generated.


rater11 = [0 if i <4 else 1 for i in rater1]
rater22 = [0 if i <4 else 1 for i in rater2]
rater33 = [0 if i <4 else 1 for i in rater3]
print(rater11)
print(rater22)
print(rater33)

print(sum(rater11))
print(sum(rater22))
print(sum(rater33))

agg = irr.aggregate_raters([rater11, rater22, rater33])

giro = np.array([rater11, rater22, rater33]).transpose()

res = irr.fleiss_kappa(irr.aggregate_raters(giro)[0], method='fleiss')
print(res)

