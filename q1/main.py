import numpy as np
n, u, v = np.array([int(i) for i in input().split(' ')])
c = np.array([int(i) for i in input().split(' ')])
l_eq = []
g_eq = []
for _ in range(u):
    l_eq += [(np.array([int(i) for i in input().split(' ')]))]
for _ in range(v):
    g_eq += [(np.array([int(i) for i in input().split(' ')]))]
rhs = (np.array([int(i) for i in input().split(' ')]))
rhs_l_eq = np.array(rhs[:u])
rhs_g_eq = np.array(rhs[u:])
l_eq = np.array(l_eq)
g_eq = np.array(g_eq)



def visualise(n, u, v, c, l_eq, g_eq, rhs_l_eq, rhs_g_eq):
    '''
    This is a function used to visualise the problem once the parameters are given
    n -> The dimensions (of x), i.e. the number of variables in the optimisation problem
    u -> The number of less than or equal to constraints
    v -> The number of greater than or equal to constraints
    l_eq -> The coefficients of each of the n variables (Shape -> (u, n))
    g_eq -> The coefficients of each of the n variables (Shape -> (v, n))
    rhs_l_eq -> The coefficients of each of the n variables (Shape -> (v))
    rhs_g_eq -> The coefficients of each of the n variables (Shape -> (u))
    '''
    print("Minimise")
    for i in range(n-1):
        print(str(c[i])+"x"+str(i+1)+" + ", end='')
    print(str(c[-1])+"x"+str(n))
    print("Such That")
    for i in range(u):
        for j in range(n-1):
            print(str(l_eq[i][j])+"x"+str(j+1)+" + ", end='')
        print(str(l_eq[i][-1])+"x"+str(n)+" | "+str(rhs_l_eq[i]))
    for i in range(v):
        for j in range(n-1):
            print(str(g_eq[i][j])+"x"+str(j+1)+" + ", end='')
        print(str(g_eq[i][-1])+"x"+str(n)+" | "+str(rhs_g_eq[i]))


# print("Before: ")
# visualise(n, u, v, c, l_eq, g_eq, rhs_l_eq, rhs_g_eq)


'''
Converting the Inequalities into equalities
'''

def convert_equalitites(n, u, v, c, l_eq, g_eq, rhs_l_eq, rhs_g_eq):
    for i in range(u):
        slack_ar = [[int(j==i)] for j in range(u+v)]
        l_eq = np.append(l_eq, slack_ar[:u], axis = 1)
        g_eq = np.append(g_eq, slack_ar[u:], axis = 1)
        c = np.append(c, [0], axis = 0)

    for i in range(v):
        slack_ar = [[int(j==(u+i))] for j in range(u+v)]
        artificial_ar = [[-1*int(j==(u+i))] for j in range(u+v)]
        if(u != 0):
            l_eq = np.append(l_eq, slack_ar[:u], axis = 1)
        g_eq = np.append(g_eq, slack_ar[u:], axis = 1)
        c = np.append(c, [0], axis = 0)
        if(u != 0):
            l_eq = np.append(l_eq, artificial_ar[:u], axis = 1)
        g_eq = np.append(g_eq, artificial_ar[u:], axis = 1)
        c = np.append(c, [0], axis = 0)
    n = n+u+2*v
    return n, u, v, c, l_eq, g_eq, rhs_l_eq, rhs_g_eq

n_init = n
n, u, v, c, l_eq, g_eq, rhs_l_eq, rhs_g_eq = convert_equalitites(n, u, v, c, l_eq, g_eq, rhs_l_eq, rhs_g_eq)

# print("\n\n\nAfter: ")
# visualise(n, u, v,c,  l_eq, g_eq, rhs_l_eq, rhs_g_eq)

# Now we have to convert into c, A, b
'''
min cTx
s.t.
Ax = b
'''
A = np.append(l_eq, g_eq, axis = 0).astype(dtype=np.float64)
b = np.append(rhs_l_eq, rhs_g_eq)




class simplex_method():
    def __init__(self, A, b, c, n, u, v):
        self.A = A
        self.b = b
        self.c = c
        self.n = n
        self.u = u
        self.v = v
        self.c_hat = None
        pass
    def phase_1(self):
        # print(np.array(list(zip(self.A, self.b)), dtype=object))
        self.c_hat = np.array([0 for i in range(self.n+self.u)] + [1, 0]*v).astype(np.float64)
        c_hat = self.c_hat
        A_hat = self.A
        # Changing the sign of the artificial variables in cost
        for i in range(self.v):
            c_hat -= A[u+i]
        basis = [self.n+i-1 for i in range(1, self.u+1)] + [self.n+self.u+i-1 for i in range(1, 2*self.v+1, 2)]
        while True:
            # select the entering variable
            j = np.argmin(c_hat)
            # select the leaving variable
            # if all the elements of the column are positive, then the problem is unbounded
            if(np.all(A_hat[:, j] >= 0)):
                print("Infeasible")
                return
            ratios = np.array([self.b[i]/self.A[i][j] if self.A[i][j] > 0 else np.inf for i in range(self.A.shape[0])])
            ratio_min = np.argmin(ratios)]
            if(ratios[ratio_min] == 0):
                print("Cycling")
                return
            entering_variable = basis[ratio_min]
            pivot = A_hat[ratio_min][j]
            # update the basis
            basis[ratio_min] = j
            # update the matrix
            A_hat[ratio_min] = A_hat[ratio_min]/pivot
            b[ratio_min] = b[ratio_min]/pivot
            for i in range(A_hat.shape[0]):
                if(i != ratio_min):
                    A_hat[i] = A_hat[i] - A_hat[i][j]*A_hat[ratio_min]
                    b[i] = b[i] - b[i]*b[ratio_min]
            c_hat = c_hat - c_hat[j]*A_hat[ratio_min]
            self.c_hat = c_hat
            if(np.all(c_hat >= 0)):
                break
        solution = [0 for i in range(self.n + self.u + 2*self.v)]
        for i in range(len(basis)):
            solution[basis[i]] = b[i]
        solution = np.array(solution)
        return solution, c_hat, basis, A_hat, b
    def phase_2(self):
        pass



    def solve(self):
        solution, c_hat, basis, A_hat, b = self.phase_1()
        print(solution)
        print("Phase 2 is a Go")
        if(np.dot(solution, c_hat) == 0):
            self.phase_2()
    def state_check(self):
        print("State Check: ")
        print(self.n, self.u, self.v)
        print(self.A, self.b)
        print(self.c_hat)

simp = simplex_method(A, b, c, n_init, u, v)
simp.solve()