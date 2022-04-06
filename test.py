import numpy as np
from torch import frac

T = 3
D = 5
M = 2



log_Ps = np.ones((M,T))
X = np.ones((T,D))

f = (np.exp(log_Ps) @ X)/np.sum(np.exp(log_Ps), axis=1)
ff=np.sum(np.exp(log_Ps), axis=1)[:, None]
print(ff.shape)



# x_mu_diff_sq = (x-mu[m]) **2
# fraction = x_mu_diff_sq/sigma[m]
# summ = -(1/2) * np.sum(fraction,-1) 
# numerator = np.exp(summ)


# # print(numerator)