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

def print_table(table, var_names, base_indices, m, iteration):
   print(f'\nIteracija: {iteration}')
   print(f'{'baze':<7}' + ''.join([f'{name:<9}' for name in var_names]) + 'B')
   print('-' * (7 + (len(var_names) + 1) * 9))
   for i in range(m):
      base_var_name = var_names[base_indices[i]]
      table_vars = ''.join([f'{var:<9.3f}' for var in table[i, :]])
      print(f'{base_var_name:<7}{table_vars}')

   print(f'{'z':<7}' + ''.join([f'{var:<9.3f}' for var in table[-1, :]]))
   print('-' * (7 + (len(var_names) + 1) * 9))

def create_table(A_matrix, B_vector, C_vector):
   m = len(A_matrix)
   n = len(A_matrix[0])

   table = np.zeros((m + 1, n + 1), dtype=float)

   table[0:m, 0:n] = A_matrix
   table[0:m, -1] = B_vector
   table[-1, 0:n] = C_vector

   base_indices = list(range(n - m, n))

   return table, m, n, base_indices

def pivot(table, pivot_col, base_indices):
   column = table[:-1, pivot_col]

   positive_rows = column > 0

   if np.any(positive_rows):
      ratios = np.full(len(column), np.inf)
      ratios[positive_rows] = table[:-1, -1][positive_rows] / column[positive_rows]

      pivot_row = np.argmin(ratios)
      pivot_row_val = table[pivot_row, pivot_col]
      base_indices[pivot_row] = pivot_col

      table[pivot_row] /= pivot_row_val

      for i in range(len(table)):
         if i == pivot_row:
            continue

         table[i] -= (table[pivot_row] * table[i, pivot_col])    
      # Maybe return pivot_row?

def solve_simplex(A_matrix, B_vector, C_vector, var_names, max_iter=10):
   table, m, n, base_indices = create_table(A_matrix, B_vector, C_vector)
   
   optimal = False

   print_table(table, var_names, base_indices, m, 0)

   for i in range(1, max_iter):
      min_z_index = np.argmin(table[-1, :-1])

      if(table[-1][min_z_index] >= 0):
         print('Optimalus sprendinys rastas')
         break
      
      pivot(table, min_z_index, base_indices)
      print_table(table, var_names, base_indices, m, i)

def main():
   a, b, c = 8.0, 10.0, 3.0
   # a, b, c = 9.0, 3.0, 4.0 # Studento numeriai


   var_names = ['x1', 'x2', 'x3', 'x4', 's1', 's2', 's3']
   A = np.array([
      [-1.0,  1.0, -1.0, -1.0, 1.0, 0.0, 0.0],
      [ 2.0,  4.0,  0.0,  0.0, 0.0, 1.0, 0.0],
      [ 0.0,  0.0,  1.0,  1.0, 0.0, 0.0, 1.0]
   ])
   B = np.array([a, b, c])
   C = np.array([2.0, -3.0, 0.0, -5.0, 0.0, 0.0, 0.0])

   solve_simplex(A, B, C, var_names)

if __name__ == '__main__':
   main()