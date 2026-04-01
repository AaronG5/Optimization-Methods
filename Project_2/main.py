import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1   = 2ab + 2bc + 2ac
# x_1 = 2ab
# x_2 = 2bc
# x_3 = 2ac
#
# 1   = x_1 + x_2 + x_3
# x_3 = 1 - x_1 - x_2
# 
# V   = abc
# x_1 * x_2 * x_3 = (2ab) * (2bc) * (2ac) = 8 * a^2 * b^2 * c^2 = 8 * (abc)^2 = 8 * V^2
# V^2 = (x_1 * x_2 * x_3) / 8
#
# f(x_1, x_2) = x_1 * x_2 * x_3 = x_1 * x_2 * (1 - x_1 - x_2)
# max(f(x_1, x_2)) = min(- f(x_1, x_2)) => 
# -f(x_1, x_2) = - (x_1 * x_2 * (1 - x_1 - x_2))

def neg_func(x_1, x_2):
   return  x_1 * x_2 * (x_1 + x_2 - 1)

def partial_deriv_x_1(x_1, x_2):
   return  x_2 * (2 * x_1 + x_2 - 1)

def partial_deriv_x_2(x_1, x_2):
   return  x_1 * (x_1 + 2 * x_2 - 1)

def grad_neg_func(x_1, x_2):
   return np.array((partial_deriv_x_1(x_1, x_2), partial_deriv_x_2(x_1, x_2)))

def line_search(x, S_i, a=0, b=1, tolerance=0.0001):
   tau = (np.sqrt(5) - 1) / 2

   L = abs(b - a)

   c = b - tau * L
   d = a + tau * L

   while L > tolerance:

      x_c = x + c * S_i
      x_d = x + d * S_i
      f_c = neg_func(x_c[0], x_c[1])
      f_d = neg_func(x_d[0], x_d[1])

      if f_c < f_d:
         b = d 
      else:
         a = c

      L = abs(b - a)
      c = b - tau * L
      d = a + tau * L

   return (a + b) / 2

def make_simplex_point(x, offset_1, offset_2):
   p = x.copy()
   p[0] = x[0] + offset_1 if 0 <= x[0] + offset_1 <= 1 else x[0] - offset_1
   p[1] = x[1] + offset_2 if 0 <= x[1] + offset_2 <= 1 else x[1] - offset_2
   return p

def gradient_descent(x, tolerance):
   x = np.array(x, dtype=float)
   step = 0.1
   
   cycle_count = 0
   func_comp_count = 0
   test_points = [x.copy()]

   while True:
      gradient = grad_neg_func(x[0], x[1])
      func_comp_count += 1

      norm = np.sqrt(gradient[0]**2 + gradient[1]**2)
      if norm < tolerance:
         break

      x = x - step * gradient
      cycle_count += 1
      test_points.append(x.copy())

   return x, neg_func(x[0], x[1]), cycle_count, func_comp_count, test_points

def steepest_descent(x, tolerance):
   x = np.array(x, dtype=float)
   
   cycle_count = 0
   func_comp_count = 0
   test_points = [x.copy()]

   while True:
      gradient = grad_neg_func(x[0], x[1])
      func_comp_count += 1

      norm = np.sqrt(gradient[0]**2 + gradient[1]**2)
      if norm < tolerance:
         break

      S_i = -(gradient / norm) # Nusileidimo krypties vektorius
      step = line_search(x, S_i)
      x = x + step * S_i

      x = np.clip(x, 0.001, 0.998)
      cycle_count += 1
      test_points.append(x.copy())

   return x, neg_func(x[0], x[1]), cycle_count, func_comp_count, test_points

def nelder_mead(x, tolerance):
   cycle_count = 0
   func_comp_count = 0
   test_points = []

   n = 2 
   c = 0.5
   gamma = 2
   beta = 0.5
   nu = -0.5

   b = c / (n * np.sqrt(2)) * (np.sqrt(n+1) - 1)
   a = b + c / np.sqrt(2)
   
   points = [
      np.array([x[0], x[1]], dtype=float),
      make_simplex_point(np.array(x, dtype=float), a, b),
      make_simplex_point(np.array(x, dtype=float), b, a)
   ]

   points.sort(key=lambda p: neg_func(p[0],p[1]))

   while True:
      points = [np.clip(p, 0.001, 0.998) for p in points]
      points_with_f = sorted(
         [(p, neg_func(p[0], p[1])) for p in points],
         key=lambda pair: pair[1]
      )
      func_comp_count += 3

      (x_l, f_l), (x_g, f_g), (x_h, f_h) = points_with_f
      points = [x_l, x_g, x_h]
      points = np.clip(points, 0.001, 0.998)

      func_values = np.array([f_l, f_g, f_h])

      test_points.append([x_l.copy(), x_g.copy(), x_h.copy()])

      spread = np.sqrt(np.sum((func_values - f_l)**2) / n)
      if spread < tolerance:
         break

      x_c = (x_l + x_g) / 2
      theta = 1
      x_naujas = x_h + (1 + theta) * (x_c - x_h)
      f_naujas = neg_func(x_naujas[0], x_naujas[1])
      func_comp_count += 1

      if f_naujas < f_l:
         theta = gamma
      elif f_naujas > f_h:
         theta = nu 
      elif f_g < f_naujas < f_h:
         theta = beta
      # elif f_l < f_naujas < f_g:
      #    theta = 1

      z = x_h + (1 + theta) * (x_c - x_h)
      
      points = [x_l, x_g, z]
      cycle_count += 1
      
   return x_l, neg_func(points[0][0], points[0][1]), cycle_count, func_comp_count, test_points

def plot_result(x_res, test_points, method_name, x_name):
   x_1_values = np.linspace(0.001, 1, 300)
   x_2_values = np.linspace(0.001, 1, 300)
   x_1, x_2 = np.meshgrid(x_1_values, x_2_values)
   z = neg_func(x_1, x_2)

   fig, ax = plt.subplots()

   contour = ax.contour(x_1, x_2, z, levels=100, cmap='viridis')
   fig.colorbar(contour, ax=ax, label='f(x1, x2)')

   if method_name == 'Deformuojamas Simpleksas':
      n_triangles = len(test_points)
      for i, triangle in enumerate(test_points):
         opacity = 0.1 + 0.9 * (i / n_triangles)
         polygon = plt.Polygon(triangle, fill=False, edgecolor='orange', alpha=opacity, linewidth=0.8)
         ax.add_patch(polygon)
      first = np.array(test_points[0])
      ax.plot(first[:, 0], first[:, 1], 'ro', linewidth=2, label='Pradinis taškas')
   else:
      points = np.array(test_points)
      ax.plot(points[:, 0], points[:, 1], '.', color='black',  markersize=4, label='Bandymo taškai')
      ax.plot(points[0, 0], points[0, 1], 'o', color='red', markersize=6, label='Pradinis taškas')

   ax.plot(x_res[0], x_res[1], '*', color='green', markersize=10, label='Minimumas') 

   ax.set_xlabel('x1')
   ax.set_ylabel('x2')
   ax.set_title(f'{method_name.replace("c", "č")} su pradiniu tašku {x_name}')
   ax.legend()
   
   method_name_replaced = method_name.replace(' ', '_')
   filepath = 'Project_2/' + method_name_replaced + '/' + method_name_replaced + '_' + x_name + '.png'
   plt.savefig(filepath, dpi=300, bbox_inches="tight")
   # plt.show()
   plt.close()

def main():
   tolerance = 0.0001
   rows = []
   for method, method_name in [
      (gradient_descent, 'Gradientinis Nusileidimas'),
      (steepest_descent, 'Greiciausias Nusileidimas'),
      (nelder_mead, 'Deformuojamas Simpleksas')
   ]:
      for x, x_name in [
         ((0, 0), 'x_0'),
         ((1, 1), 'x_1'),
         ((3/10, 4/10), 'x_m')
      ]:
         x_res, z_res, cycle_count, func_comp_count, test_points = method(x, tolerance)
         plot_result(x_res, test_points, method_name, x_name)
         rows.append({
            'Method': method_name,
            'Starting_point': x_name,
            'x1_x2': x_res,
            'Minimum_value': round(z_res, 6),
            'Cycle_count': cycle_count,
            'Func_comp_count': func_comp_count
         })
   df = pd.DataFrame(rows)
   csv_filepath = 'Project_2/Gradientu_Lentele.csv'
   df.to_csv(csv_filepath)

if __name__ == "__main__":
   main()