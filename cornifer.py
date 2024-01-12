import cvxpy as cp
import numpy as np
import pandas as pd
import os
import time
import argparse
import math

'''
Processes each metro in a topology, and finds:
alpha: latency ask
mu: time ask
C: clients at each metro 
default_pops: geographically closest pop at each metro 
L: 2D matrix for each metro*pop
S: 2D matrix for each metro*pop
'''

base_dir = os.getcwd()

def process_topology(topology_path, SLA_path, window):

   topology_df = pd.read_csv(topology_path)
   metros = []
   all_unique_pops = []
   unique_metro_pops = pd.read_csv(str(base_dir)+"/metro_default_pops.csv")
   

   alpha = []
   mu = []
   C = []
   default_pops = []
   for index, row in topology_df.iterrows():
      metro = row["metro"]
      metros.append(metro)
      alpha.append(row["latency"])
      mu.append(row["time"])
      C.append(row["clients"])
      default_pops.append(row["default pop"])
      all_pops = unique_metro_pops.loc[unique_metro_pops["metro"] == str(metro)]["pop choices"]
      pops = str(all_pops.values).replace("[", "")
      pops = pops.replace("'", "")
      pops = pops.replace("]", "")
      pops = pops.replace("\"", "")
      pop = list(pops.split(" "))
      all_unique_pops.extend(pop)

   all_unique_pops = list(set(all_unique_pops))

   n = len(all_unique_pops)
   m = len(metros)
   L = np.empty(shape=(n, m))
   S = np.empty(shape=(n, m))
  
   for i in range(len(metros)):
      sla = pd.read_csv(str(SLA_path)+str(metros[i])+".csv")
      
      for j in range(len(all_unique_pops)):
         pop = all_unique_pops[j]
         pop_rows = sla.loc[sla["pop"] == str(pop)]
         if(len(pop_rows) == 0):
            L[j][i] = 9999
            S[j][i] = 0
         else:
            S[j][i] = window
            lat = pop_rows.loc[pop_rows["window"] == window]["latency"]
            L[j][i] = lat

   return L, S, np.array(alpha), np.array(mu), np.array(C), all_unique_pops, metros, default_pops


def optimization_with_hub_v2(L, S, alpha, mu, C, P, M, default_pops, K, output_file, start_time, sla_constraints = True, 
                              optimize_k = True, fixed_k = False):

   beta = 1000
   constraints = []

   n = len(P)
   m = len(M)

   #auxiliary variable 
   U_M = cp.Variable((m,n), integer = True)# boolean = True)
   constraints.append(U_M >= 0)
   constraints.append(U_M <= 1)



   row_constraint_1 = [0] * m
   #row_constraint_2 = [0] * m
  
   for i in range(m):
      row_constraint_1[i] = 0  

      #SLO constraints: optimal
      if(sla_constraints):
         constraints.append(U_M[i]@L[:,i] <= alpha[i])
         constraints.append(U_M[i]@S[:,i] >= mu[i])

      #k-optimal and  l-optimal
      elif(not sla_constraints):
         constraints.append(U_M[i]@S[:,i] >= mu[i])

      
      for j in range(n):
         row_constraint_1[i] = row_constraint_1[i] + U_M[i][j]

      constraints.append(row_constraint_1[i] == 1)
   
  
   unique_pops = [] 

   for i in range(n):
      unique_pops.append(cp.Variable(integer=True))
      constraints.append(unique_pops[i]>=0)
      constraints.append(unique_pops[i]<=1)
   
   
   #Unique pops
   for i in range(n):
      constraints.append(unique_pops[i] - cp.max(U_M[:,i]) >= 0)

    # optimal and k-optimal
   if(optimize_k):
      total_unique_pops = 0
      for k in range(n):
         total_unique_pops = total_unique_pops + unique_pops[k]
      
      #if(sla_constraints):
      if(fixed_k):
         constraints.append(total_unique_pops == K)
      else:
         constraints.append(total_unique_pops <= K)

   #objective to minimize sum of (selected metro row from U_M * selected pop columns from the L for the metro)
   sum = 0
   for i in range(m):
         sum = sum + ((U_M[i]@L[:,i]) * C[i])
   objective = cp.Minimize(sum)

  
   prob = cp.Problem(objective, constraints)
   prob.solve(verbose=True)  # Returns the optimal value.
  
   if(prob.status == cp.OPTIMAL):
      output = open(str(output_file), "w")
      output.write(",".join(["metro", "clients", "default pop", 
                              "pop chosen", "lat chosen", "weighted latency", "execution_time"]) + "\n")
 
    
      pop_indices = [-1] * m
      for i in range(m):
         for j in range(n):
            if(int(U_M[i][j].value) == 1):
               pop_indices[i] = j
               break

      for i in range(m):
         x = pop_indices[i]
         output.write(",".join([str(M[i]), str(C[i]), str(default_pops[i]), 
                        str(P[x]),  str(L[x][i]), str(prob.value),
                        str(time.time() - start_time)]) + "\n")

      output.close()

   return prob




'''
Finds a solution for optimal case by performing binary serach
'''      
def find_optimal_solution(start, end, last_optimal_k, 
                        L,S, alpha, mu, C, P, M, default_pops, output_file, start_time, sla_constraints, optimize_k):
   if(end >= start): 
      mid = int(start + (end - start) / 2)

      prob = optimization_with_hub_v2(L,S, alpha, mu, C, P, M, default_pops, mid, str(output_file), start_time, sla_constraints, optimize_k, False)

      if(prob.status != cp.OPTIMAL):
         if(last_optimal_k != -1):
               start = mid + 1
               end = last_optimal_k
         else:
               start = mid +1
         return find_optimal_solution(start, end, last_optimal_k, 
                              L,S, alpha, mu, C, P, M, default_pops, output_file, start_time, sla_constraints, optimize_k)
      
      else:
         if(mid == last_optimal_k):
               return last_optimal_k

         if(last_optimal_k == -1):
               end = mid - 1
               last_optimal_k = mid
         
         if(last_optimal_k != -1 and mid < last_optimal_k):
               end = mid -1
               last_optimal_k = mid

         if(last_optimal_k != -1 and mid > last_optimal_k):
               end = last_optimal_k
         
         return find_optimal_solution(start, end, last_optimal_k, 
                              L,S, alpha, mu, C, P, M, default_pops, output_file, start_time, sla_constraints, optimize_k)
   
   return last_optimal_k

'''
Finds optimal placement for a toplogy based on the mode selected.
'''
def find_placements(size, i, topology_path, result_path, sla_path, window, sla_constraints, optimize_k, fixed_k, mean_k_file):
   beta = 1000
   
   if(fixed_k):
      mean_k_df = pd.DataFrame()
      mean_k_df = pd.read_csv(mean_k_file)

   topology = "topology_"+str(size)+"_"+str(i)
   f = topology_path + "topology_"+str(size)+"_"+str(i)+".csv"

   output_file = result_path + "topology_"+str(size)+"_"+str(i)+".csv"
   start_time = time.time()
   L, S, alpha, mu, C, P, M, default_pops = process_topology(str(f), str(sla_path), window)

   #slo-optimal
   if(sla_constraints and optimize_k and not fixed_k):
      optimal_k = find_optimal_solution(0, size, -1,
                                 L,S, alpha, mu, C, P, M, default_pops, str(output_file), start_time, sla_constraints, optimize_k)

   #l-optimal: sla_constraints = false, optimize_k = false
   elif(not sla_constraints and not optimize_k and not fixed_k):
      prob = optimization_with_hub_v2(L,S, alpha, mu, C, P, M, default_pops, 1, str(output_file), start_time, sla_constraints, optimize_k, False)
   
   #k-optimal: sla_constraints = false, optimize_k = true
   elif(optimize_k and not sla_constraints and not fixed_k):
      K = 1
      prob = optimization_with_hub_v2(L,S, alpha, mu, C, P, M, default_pops, K, str(output_file), start_time, sla_constraints, optimize_k, False)
      while(prob.status != cp.OPTIMAL):
         K = K + 1
         prob = optimization_with_hub_v2(L,S, alpha, mu, C, P, M, default_pops, K, str(output_file), start_time, sla_constraints, optimize_k, False)
      
   #mean-k
   elif(optimize_k and fixed_k and not sla_constraints):
      mean_k = mean_k_df.loc[mean_k_df['topology'] == topology]["mean_k"].values[0]
      prob = optimization_with_hub_v2(L,S, alpha, mu, C, P, M, default_pops, mean_k, str(output_file), start_time, sla_constraints, optimize_k, fixed_k)
   
   return


def find_mean_k(result_path, topology):
   l_optimal_file = result_path + "l_optimal/"+str(topology)+".csv"
   k_optimal_file = result_path + "k_optimal/"+str(topology)+".csv"
   df_l_optimal = pd.read_csv(l_optimal_file)
   df_k_optimal = pd.read_csv(k_optimal_file)

   max_k = df_l_optimal["pop chosen"].nunique()
   min_k = df_k_optimal["pop chosen"].nunique()
   mean_k = math.floor((min_k+max_k)/2)

   out_dir = result_path + "mean_k_values/"

   if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)

   output_file =  open(out_dir + str(topology) + "_mean_k.csv", "w")
   output_file.write(",".join(["topology", "min_k", "max_k", "mean_k"]) + "\n")


   output_file.write(",".join([str(topology), str(min_k), str(max_k), str(mean_k)]) + "\n")

   output_file.close()

   return 

'''
FInds solution for each topology
'''
def run_optimization(result_path, size, topo_n):
   modes = ["k_optimal", "l_optimal", "mean_k"]
   topology = "topology_"+str(size)+"_"+str(topo_n)

   mean_k_file = ""

   #set flags for each mode
   for mode in modes: 
      sla_constraints = True
      optimize_k = True
      fixed_k = False
      #optimize_h = True
      if(mode == "optimal"):
         sla_constraints = True
         optimize_k = True
         mode_result_path = result_path  + "slo_optimal/"
         #optimize_h = False
      elif(mode == "k_optimal"):
         sla_constraints = False
         optimize_k= True
         mode_result_path = result_path  + "k_optimal/"
         #optimize_h = False
      elif(mode == "l_optimal"):
         optimize_k = False
         sla_constraints = False
         mode_result_path = result_path  + "l_optimal/"

      elif(mode == "mean_k"):
         find_mean_k(result_path, topology)
         sla_constraints = False
         optimize_k = True
         fixed_k = True
         mode_result_path = result_path  + "mean_k/"
         mean_k_file = result_path + "mean_k_values/"+str(topology) + "_mean_k.csv"
     

      
      if(not os.path.exists(mode_result_path)):
            os.mkdir(mode_result_path)

      topology_path = base_dir + "/topologies/"

      sla_path = base_dir + '/metro-pop-lat/' 

      find_placements(size, topo_n, topology_path,
                  mode_result_path,
                  sla_path, 90, sla_constraints, optimize_k, fixed_k, mean_k_file)


if __name__=="__main__":
   
   results_path = base_dir + "/results/"

   if(not os.path.exists(results_path)):
      os.mkdir(results_path)

   parser = argparse.ArgumentParser(description="Usage: python cornifer.py -s <topology_size> -n <topology_number>",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
   parser.add_argument("-s", "--size", help="Topology size", choices=[5, 10, 25, 50, 75], type=int, required= True)
   parser.add_argument("-n", "--topo_number", help="Topology number", choices=list(range(1,11)), type=int, required= True)

   args = parser.parse_args()
   size = args.size 
   topo_n = args.topo_number

   
   run_optimization(results_path, size, topo_n)