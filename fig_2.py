import numpy as np
import matplotlib.pyplot as pl

def lower_bound(sigma, c):
    eta = np.exp(-sigma**2/2)
    return eta**2/(1-eta**2*np.cosh(sigma**2*c)+2*eta**2*sum(np.sinh(sigma**2*c**n) for n in range(1, max(3000,int(1000/np.log(abs(c)))))))

def upper_bound(sigma, c):
    fisher_uncorrelated = 1/(np.exp(sigma**2*(1-abs(c))/(1+abs(c))) - 1)
    if c >= 0:
        fisher_correlated = (1-c**2)/(4*c*sigma**2)
        return fisher_correlated*fisher_uncorrelated/(fisher_correlated + fisher_uncorrelated)
    else: return fisher_uncorrelated

def upper_bound_old(sigma, c):
    return (1-c)/(1+c)*sigma**-2

sigma_values = [(0.5**0.5, 'tab:blue', '0.5'), (1, 'tab:red', '1.0'), (2.0**0.5, 'tab:green', '2.0')]
c_ran = np.linspace(-0.99, 0.99, 100)


# auxilary plots to generate legend values
for _, color, _ in sigma_values: pl.scatter([],[], c = color)    
pl.plot([2],[200], c = 'black')
pl.plot([2],[200], '--', c = 'black')
pl.plot([2],[200], ':', c = 'black')

    
for sigma, color, _ in sigma_values:
    up1 = [upper_bound(sigma, c) for c in c_ran]
    up2 = [upper_bound_old(sigma, c) for c in c_ran]
    pl.plot(c_ran, [lower_bound(sigma, c) for c in c_ran], ':', c = color)
    pl.plot(c_ran, up1, '--', c = color)
    pl.plot(c_ran, up2, c = color)
    pl.fill_between(c_ran, [lower_bound(sigma, c) for c in c_ran], [min(a,b) for a,b in zip(up1, up2)], color = color, alpha = 0.1)
    
pl.rcParams['font.serif'] = "Times"
pl.rcParams['font.size'] = 12
pl.xlabel(r"Correlation parameter $c$")
pl.ylabel(r"$F_Q^{(N)}/N$")
pl.xlim([-1,1])
pl.yscale('log')
pl.ylim([0.1,100])

pl.legend([r"$\sigma^2 = " + sigma +"$" for _, _, sigma in sigma_values] + ['Standard CS bound', 'Refined CS bound', 'SS strategy'])
# pl.savefig('plot1.pdf', bbox_inches = 'tight')
pl.show()