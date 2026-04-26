import matplotlib.pyplot as plt
import numpy as np

# 1 = 2ab + 2bc + 2ac
# 1 = 2 * (ab + bc + ac)
# f(x)  = a * b * c   # Tūris
# -f(x) = - a * b * c # Neigiama funkcija skirta minimizavimui
#
# g(x)  = 2 * (ab + bc + ac) - 1 = 0 # Lygybinio apribojimo funkcija
# h1(x) = -a ≤ 0 
# h2(x) = -b ≤ 0
# h3(x) = -c ≤ 0

def f(X: np.ndarray) -> float:
   x1, x2, x3 = X
   return - x1 * x2 * x3

def g(X: np.ndarray) -> float:
   x1, x2, x3 = X
   return 2 * (x1*x2 + x2*x3 + x1*x3) - 1

def h1(X: np.ndarray) -> float:
   return -X[0]

def h2(X: np.ndarray) -> float:
   return -X[1]

def h3(X: np.ndarray) -> float:
   return -X[2]

def penalty_f(X: np.ndarray, r: float) -> float:
   eq_penalty = g(X)**2
   ineq_penalty = max(0, h1(X))**2 + max(0, h2(X))**2 + max(0, h3(X))**2
   return f(X) + (1/r) * (eq_penalty + ineq_penalty)

def penalty_mult_impact(X_0, X_1, X_m):
   plt.figure(figsize=(6, 4))
   plt.title('Baudos funkcijos reikšmės su skirtingomis X vertėmis, r -> 0')
   plt.xlim((0.51, 0.0001))
   plt.xlabel('r')
   plt.ylim((-1.01, 500))
   plt.ylabel('P(X, r)')
   r_range = np.linspace(1, 0.0001, 500)
   penalty_X_0 = [penalty_f(X_0, r) for r in r_range]
   penalty_X_1 = [penalty_f(X_1, r) for r in r_range]
   penalty_X_m = [penalty_f(X_m, r) for r in r_range]
   plt.plot(r_range, penalty_X_0, color ='red', label='f(X_0)')
   plt.plot(r_range, penalty_X_1, color ='blue', label='f(X_1)')
   plt.plot(r_range, penalty_X_m, color ='green', label='f(X_m)')
   plt.legend()
   plt.show()

def main():
   X_0 = (0, 0, 0)
   X_1 = (1, 1, 1)
   X_m = (9/10, 3/10, 4/10)

   for X, name in ((X_0, 'X_0'), (X_1, 'X_1'), (X_m, 'X_m')):
      print(f'{name}: {X} \nf({name}: {f(X)} \ng({name}): {g(X)} \
            \nh1({name}): {h1(X)} \nh2({name}): {h2(X)} \nh2({name}): {h2(X)}')

   penalty_mult_impact(X_0, X_1, X_m)

if __name__ == "__main__":
   main()