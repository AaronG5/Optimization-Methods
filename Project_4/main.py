# min (2*x_1 - 3*x_2 - 5*x_4)
# Apribojimai:
# 1. -x_1 + x_2 - x_3 - x_4 ≤ 8
# 2. 2*x_1 + 4*x_2 ≤ 10
# 3. x_3 + x_4 ≤ 3
# 4. x_i ≥ 0

# a = 9
# b = 3
# c = 4

import numpy as np

def print_table(table, var_names, base_indices, m, iteration=0):
   print(f'\nIteracija: {iteration}')
   print(f'{'baze':<7}' + ''.join([f'{name:<9}' for name in var_names]) + 'B')
   print('-' * (7 + (len(var_names) + 1) * 9))
   for i in range(m):
      base_var_name = var_names[base_indices[i]]
      table_vars = ''.join([f'{var:<9.3f}' for var in table[i, :]])
      print(f'{base_var_name:<7}{table_vars}')

   print(f'{'z':<7}' + ''.join([f'{var:<9.3f}' for var in table[-1, :]]))
   print('-' * (7 + (len(var_names) + 1) * 9))

def pivot():
   pass

def create_table(A_matrix, B_vector, C_vector):
   m = len(A_matrix)
   n = len(A_matrix[0])

   table = np.zeros((m + 1, n + 1), dtype=float)

   table[0:m, 0:n] = A_matrix
   table[0:m, -1] = B_vector
   table[-1, 0:n] = C_vector

   base_indices = list(range(n - m, n))

   return table, m, n, base_indices

def solve_simplex(A_matrix, B_vector, C_vector, var_names):
   table, m, n, base_indices = create_table(A_matrix, B_vector, C_vector)
   
   print_table(table, var_names, base_indices, m)


def main():
   var_names = ['x1', 'x2', 'x3', 'x4', 's1', 's2', 's3']
   A = np.array([
      [-1.0,  1.0, -1.0, -1.0, 1.0, 0.0, 0.0],
      [ 2.0,  4.0,  0.0,  0.0, 0.0, 1.0, 0.0],
      [ 0.0,  0.0,  1.0,  1.0, 0.0, 0.0, 1.0]
   ])
   B = np.array([8.0, 10.0, 3.0])
   C = np.array([2.0, -3.0, 0.0, -5.0, 0.0, 0.0, 0.0])

   solve_simplex(A, B, C, var_names)

if __name__ == '__main__':
   main()