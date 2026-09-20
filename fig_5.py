import numpy as np
import matplotlib.pyplot as pl

def removeDuplicates(indices):
    indices.sort()
    to_delete = set()
    for i in range(len(indices) - 1):
        if indices[i] == indices[i + 1] and i not in to_delete:
            to_delete |= {i, i + 1}
    return [x for index, x in enumerate(indices) if index not in to_delete]

def ter_mom(indices, c):
    n = len(indices)
    result = c**sum(indices[i]*(-1)**(i % 2 == 0) for i in range(n))
    for i in range(1, n//2): result *= (1 + c**(2*(indices[2*i] - indices[2*i - 1])))
    return result/2**(n//2)*(4*eta*(1-eta))**(n//2)

def ter_der(n, c, eta):
    if n == 1: return eta
    result = (n-1)*(2*eta-1)*(c/2)**((n+1)/2)*(1+c**2)**((n-3)/2) + 2*(eta + c**2*(eta-1))*(c/2)**((n-1)/2)*(1+c**2)**((n-3)/2) + (n-3)/2*(eta*(1+c**4)+2*c**2*(eta-1))*(c/2)**((n-1)/2)*(1+c**2)**((n-5)/2)
    return result*(4*eta*(1-eta))**((n-1)/2)

def bin_mom(indices, c):
    return c**sum(indices[i]*(-1)**(i % 2 == 0) for i in range(len(indices)))*(1-eta**2)**(len(indices)/2)

def bin_der(n, c, eta):
    return eta*((n+1)+(n-1)*c)/2*c**((n-1)/2)*(1-eta**2)**((n-1)/2)
    
def precision(c, eta, n, mom, der):
    s = range(1, 2*n, 2)
    K = np.zeros((n,n))
    v = np.zeros(n)
    for i in range(n):
        l = s[i]
        v[i] = der(l, c, eta)
        for j in range(i+1):
            k = s[j]
            for shift in range((l+k)//2):
                indices = removeDuplicates(list(range(-(l//2),(l//2)+1))+list(range(-(k//2)+shift,(k//2)+1+shift)))
                K[i,j] += (1 if shift == 0 else 2)*mom(indices, c)
            K[i,j] += 2*mom(list(range(l+k)), c)/(1-c)
            K[j,i] = K[i,j]
    return ((v @ np.linalg.solve(K, v))**(-1)-1)**-1

pl.rcParams['font.size'] = 12
c_ran = np.linspace(-0.999, 0.999, 200)

for c in ('tab:blue', 'tab:orange', 'tab:green'): pl.scatter([2],[2000], c = c)
pl.plot([2], [2], c = 'black')
pl.plot([2], [2], '--', c = 'black')
for sigma,c in zip((0.5, 1, 2), ('tab:blue', 'tab:orange', 'tab:green')):
    eta = np.exp(-sigma/2)
    pl.plot(c_ran, [precision(c, eta, 1, ter_mom, ter_der) for c in c_ran], '--', c = c)
    pl.plot(c_ran, [precision(c, eta, 2, ter_mom, ter_der) for c in c_ran], '-', c = c)

pl.legend([r"$\sigma^2 = 0.5$", r"$\sigma^2 = 1.0$", r"$\sigma^2 = 2.0$", r"$\vec{X}$ estimation", r"$J_y$ estimation"])
pl.xlabel(r"Correlation parameter $c$")
pl.ylabel(r"$\text{Var}(\hat\theta)^{-1}/N$")
pl.xlim([-1,1])
pl.ylim([0.02,50])
pl.yscale('log')
# pl.savefig('plot.pdf', bbox_inches = 'tight')
pl.show()
























