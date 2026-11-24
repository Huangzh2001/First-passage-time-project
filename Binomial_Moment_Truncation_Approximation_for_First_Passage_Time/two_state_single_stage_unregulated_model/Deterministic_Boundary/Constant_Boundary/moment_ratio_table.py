from math import exp, factorial, comb

def raw_ratio(gamma, m, N):
    return exp(-gamma*m) * sum((gamma*m)**r / factorial(r)
                              for r in range(N+1))

def binomial_ratio(gamma, m, N):
    return exp(-gamma*m) * sum(comb(m, r)*(exp(gamma)-1)**r
                              for r in range(min(m, N)+1))

print("gamma  m    Raw(N=8)       BM(N=8)        Raw(N=16)      BM(N=16)")
for gamma in [0.5, 2, 8]:
    for m in [7, 8, 9]:
        print(f"{gamma:5g} {m:2d} "
              f"{raw_ratio(gamma, m, 8):14.6g} "
              f"{binomial_ratio(gamma, m, 8):14.6g} "
              f"{raw_ratio(gamma, m, 16):14.6g} "
              f"{binomial_ratio(gamma, m, 16):14.6g}")
