import matplotlib.pyplot as plt
import numpy as np
import os

# 1 = 2ab + 2bc + 2ac
# 1 = 2 * (ab + bc + ac)
# f(x)  = a * b * c   # Tūris
# -f(x) = - a * b * c # Neigiama funkcija skirta minimizavimui
#
# g(x)  = 2 * (ab + bc + ac) - 1 = 0 # Lygybinio apribojimo funkcija
# h1(x) = -a ≤ 0 
# h2(x) = -b ≤ 0
# h3(x) = -c ≤ 0

def f(X: np.ndarray) -> float: # Negative function
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
   eq = g(X)**2
   ineq = max(0, h1(X))**2 + max(0, h2(X))**2 + max(0, h3(X))**2
   return f(X) + (1/r) * (eq + ineq)

def penalty_mult_impact(X_values_and_names):
   plt.figure(figsize=(6, 4))
   plt.title('Baudos funkcijos reikšmės su skirtingais X, kai r -> 0')
   plt.xlim((0.51, 0))
   plt.xlabel('r')
   plt.ylim((-1.01, 500))
   plt.ylabel('P(X, r)')
   r_range = np.linspace(1, 0.0001, 500)
   for X, name in X_values_and_names:
      penalty_X = [penalty_f(X, r) for r in r_range]
      plt.plot(r_range, penalty_X, label=f'f({name})')
   plt.legend()

   save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baudos_funcijos.png')
   plt.savefig(save_dir, dpi=300)
   plt.show()

def make_initial_simplex(X, c=0.5):
   n = len(X)
   b = c / (n * np.sqrt(2)) * (np.sqrt(n+1) - 1)
   a = b + c / np.sqrt(2)

   simplex = [X.copy()]
   for i in range(n):
      p = X.copy()
      for j in range(n):
         p[j] += a if i == j else b
      simplex.append(p)
   return simplex

def nelder_mead(X_start, r, tolerance, max_iter=10000):
   n = len(X_start)
   cycle_count = 0
   func_comp_count = 0

   gamma = 2.0
   beta = 0.5
   nu = -0.5

   func = lambda X: penalty_f(X, r)

   points = make_initial_simplex(X_start.copy())
   points_with_f = ([(p, func(p)) for p in points])
   func_comp_count += len(points)

   while True:
      points_with_f.sort(key=lambda pair: pair[1])
      points = np.array([pw[0] for pw in points_with_f])
      func_values = np.array([pw[1] for pw in points_with_f])

      x_l, f_l = points_with_f[0]  # geriausias taškas
      x_g, f_g = points_with_f[-2] # antras blogiausias
      x_h, f_h = points_with_f[-1] # blogiausias

      spread = np.sqrt(np.sum((func_values - f_l)**2) / n)
      if spread < tolerance or cycle_count >= max_iter:
         break

      x_c = np.mean(points[:-1], axis=0)
      theta = 1
      x_ref = x_c + (x_c - x_h)
      f_ref = func(x_ref)
      func_comp_count += 1

      if f_l < f_ref < f_g:
         points_with_f = points_with_f[:-1] + [(x_ref, f_ref)]
         continue

      if f_ref < f_l:
         theta = gamma
      elif f_ref > f_h:
         theta = nu 
      elif f_g < f_ref < f_h:
         theta = beta

      z = x_h + (1 + theta) * (x_c - x_h)
      f_z = func(z)
      func_comp_count += 1

      points_with_f = points_with_f[:-1] + [(z, f_z)]
      cycle_count += 1

   return x_l, f_l, cycle_count, func_comp_count

def penalty_method(X_start, r_start=1.0, r_factor=0.1, n_outer=2, tolerance=1e-8):
   X = X_start.copy()
   r = r_start
   total_cycles = 0
   total_evals = 0

   for _ in range(n_outer):
      X, P_val, cycles, evals = nelder_mead(X, r, tolerance)
      total_cycles += cycles
      total_evals += evals
      r *= r_factor

   return X, total_cycles, total_evals

def main():
   X_0 = np.array([0.001, 0.001, 0.001])
   X_1 = np.array([1.0, 1.0, 1.0])
   X_m = np.array([9/10, 3/10, 4/10])
   X_ats = np.array([0.41, 0.41, 0.41])

   X_values_and_names = [(X_0, 'X_0'), (X_1, 'X_1'), (X_m, 'X_m')]

   for X, name in X_values_and_names:
      print(f'{name}: {X} \
            \n - f({name})  = {f(X)} \
            \n - g({name})  = {g(X)} \
            \n - h1({name}) = {h1(X)} \
            \n - h2({name}) = {h2(X)} \
            \n - h3({name}) = {h3(X)}\n')
   
   for X, name in X_values_and_names:
      X_opt, cycles, evals = penalty_method(X)
      print(f'Pradinis taškas {name}: \
            \n - X sprendinys = {X_opt} \
            \n - Minimumo įvertis = {-f(X_opt):.8f} \
            \n - Atlikta žingsnių = {cycles} \
            \n - Funkcijų skaičiavimų skaičius = {evals}\n')

   X_values_and_names.append((X_ats, 'X_ats'))
   penalty_mult_impact(X_values_and_names)

if __name__ == "__main__":
   main()