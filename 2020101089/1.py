from re import L
import numpy as np
n, u, v = np.array([int(i) for i in input().split(' ')])
c = np.array([float(i) for i in input().split(' ')]).astype(dtype=np.float64)
l_eq = []
g_eq = []
for _ in range(u):
    l_eq += [(np.array([float(i) for i in input().split(' ')]))]
for _ in range(v):
    g_eq += [(np.array([float(i) for i in input().split(' ')]))]
rhs = (np.array([float(i) for i in input().split(' ')]))
rhs_l_eq = np.array(rhs[:u]).astype(dtype=np.float64)
rhs_g_eq = np.array(rhs[u:]).astype(dtype=np.float64)
l_eq = np.array(l_eq).astype(dtype=np.float64)
g_eq = np.array(g_eq).astype(dtype=np.float64)



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
    # print(v)
    for i in range(u):
        slack_ar = [[int(j==i)] for j in range(u+v)]
        l_eq = np.append(l_eq, slack_ar[:u], axis = 1)
        if(v != 0):
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
if(v!= 0 and u != 0):
    A = np.append(l_eq, g_eq, axis = 0).astype(dtype=np.float64)
    b = np.append(rhs_l_eq, rhs_g_eq).astype(dtype=np.float64)
elif(v == 0):
    A = l_eq
    b = rhs_l_eq
elif(u == 0):
    A = g_eq
    b = rhs_g_eq




class simplex_method():
    def __init__(self, A, b, c, n, u, v):
        self.A = A
        self.b = b
        self.c = c
        self.n = n
        self.u = u
        self.c_init = c
        self.v = v
        self.c_hat = None
        pass
    def phase_1(self):
        self.c_hat = np.array([0 for i in range(self.n+self.u)] + [-1, 0]*v).astype(np.float64)
        c_hat = self.c_hat
        A_hat = self.A
        # Changing the sign of the artificial variables in cost
        for i in range(self.v):
            c_hat += A[u+i]
        basis = [self.n+i-1 for i in range(1, self.u+1)] + [self.n+self.u+i-1 for i in range(1, 2*self.v+1, 2)]
        cycling = 0
        while True:
            if(np.all(c_hat <= 0)):
                break
            # print(A_hat, b, end = "\n\n")
            # print(c_hat)
            # print(basis, end = "\n\n")
            # select the entering variable
            # j = np.argmax(c_hat)

            j = np.argmin([c_hat[i] if c_hat[i] > 0 else np.inf for i in range(c_hat.shape[0])])
            # j = np.argmin(c_hat[(np.where(c_hat > 0))])
            # select the leaving variable
            # if all the elements of the column are positive, then the problem is unbounded in phase_1 problem
            if(np.all(A_hat[:, j] <= 0)):
                print("Infeasible")
                return
            ratios = np.array([self.b[i]/A_hat[i][j] if A_hat[i][j] > 0 else np.inf for i in range(A_hat.shape[0])])
            ratio_min = np.argmin(ratios)
                
            leaving_variable = basis[ratio_min]
            pivot = A_hat[ratio_min][j]
            basis[ratio_min] = j
            A_hat[ratio_min] = A_hat[ratio_min]/pivot
            b[ratio_min] = b[ratio_min]/pivot
            for i in range(A_hat.shape[0]):
                if(i != ratio_min):
                    b[i] = b[i] - A_hat[i][j]*b[ratio_min]
                    A_hat[i] = A_hat[i] - A_hat[i][j]*A_hat[ratio_min]
            c_hat = c_hat - c_hat[j]*A_hat[ratio_min]
            self.c_hat = c_hat

        solution = [0 for i in range(self.n + self.u + 2*self.v)]
        for i in range(len(basis)):
            solution[basis[i]] = b[i]
        solution = np.array(solution)
        return solution, c_hat, basis, A_hat, b
    


    def phase_2(self, basis, A_hat, b):
        # removing the artificial variables
        A_hat = np.delete(A_hat, [self.n+self.u+2*i for i in range(self.v)], axis = 1) 
        self.c = np.delete(c, [self.n+self.u+2*i for i in range(self.v)])
        for i in range(len(basis)):
            if(basis[i] > self.n+self.u):
                basis[i] = (self.n+self.u+basis[i])//2

        # modifyiîng the cost variable to get zeroes in basic variables
        c_new = np.zeros(self.c.shape[0])
        for i in range(len(c_new)):
            c_new[i] = np.dot(self.c[basis],A_hat[:, i])
        for i in range(len(basis)):
            c_new[basis[i]] = 0
        prev_solution = np.array([np.inf for i in range(self.n + self.u + 2*self.v)])
        cycling = 0
        while True:
            if(np.all(c_new <= 0)):
                break
            j = np.argmax(c_new)
            if(cycling):
                j = np.argmin([c_new[i] if c_new[i] > 0 else np.inf for i in range(c_new.shape[0])])
                j = np.min(np.where(self.c < 0))
            if(np.all(A_hat[:, j] <= 0)):
                print("Unbounded")
                return
            ratios = np.array([b[i]/A_hat[i][j] if A_hat[i][j] > 0 else np.inf for i in range(A_hat.shape[0])])
            ratio_min = np.argmin(ratios)
            leaving_variable = basis[ratio_min]
            pivot = A_hat[ratio_min][j]
            # update the basis
            basis[ratio_min] = j
            # update the matrix
            A_hat[ratio_min] = A_hat[ratio_min]/pivot
            b[ratio_min] = b[ratio_min]/pivot
            for i in range(A_hat.shape[0]):
                if(i != ratio_min):
                    b[i] = b[i] - A_hat[i][j]*b[ratio_min]
                    A_hat[i] = A_hat[i] - A_hat[i][j]*A_hat[ratio_min]
            c_new = c_new - c_new[j]*A_hat[ratio_min]

            solution = [0 for i in range(self.n + self.u + 2*self.v)]
            for i in range(len(basis)):
                solution[basis[i]] = b[i]
            solution = np.array(solution)
            if(np.all(solution == prev_solution)):
                cycling = 1
            prev_solution = solution

        solution = [0 for i in range(self.n + self.u + 2*self.v)]
        for i in range(len(basis)):
            solution[basis[i]] = b[i]
        solution = np.array(solution)
        print("{:.7f}".format(np.dot(solution, self.c_init)))
        for i in range(self.n):
            print("{:.7f}".format(solution[i]), end = " ")


    def temp_phase_2(self, basis, A_hat, b):
        # removing the artificial variables
        
        A_hat = np.delete(A_hat, [self.n+self.u+2*i for i in range(self.v)], axis = 1) 
        self.c = np.delete(c, [self.n+self.u+2*i for i in range(self.v)])

        # modifyiîng the cost variable to get zeroes in basic variables
        c_new = np.zeros(self.c.shape[0])
        for i in range(len(c_new)):
            c_new[i] = np.dot(self.c[basis],A_hat[:, i])
        for i in range(len(basis)):
            c_new[basis[i]] = 0
        prev_solution = np.array([np.inf for i in range(self.n + self.u + 2*self.v)])
        c_new = -self.c_init
        cycling = 0
        while True:
            if(np.all(c_new <= 0)):
                # print(c_new)
                break
            # j = np.argmin([c_new[i] if c_new[i] > 0 else np.inf for i in range(c_new.shape[0])])
            j = np.argmax(c_new)
            if(cycling == 1):
                j = np.min(np.where(self.c < 0))
                j = np.argmin([c_new[i] if c_new[i] > 0 else np.inf for i in range(c_new.shape[0])])
            if(np.all(A_hat[:, j] <= 0)):
                print("Unbounded")
                return
            ratios = np.array([b[i]/A_hat[i][j] if A_hat[i][j] > 0 else np.inf for i in range(A_hat.shape[0])])
            ratio_min = np.argmin(ratios)
            leaving_variable = basis[ratio_min]
            pivot = A_hat[ratio_min][j]
            # update the basis
            basis[ratio_min] = j
            # update the matrix
            A_hat[ratio_min] = A_hat[ratio_min]/pivot
            b[ratio_min] = b[ratio_min]/pivot
            for i in range(A_hat.shape[0]):
                if(i != ratio_min):
                    b[i] = b[i] - A_hat[i][j]*b[ratio_min]
                    A_hat[i] = A_hat[i] - A_hat[i][j]*A_hat[ratio_min]
            c_new = c_new - c_new[j]*A_hat[ratio_min]

            solution = [0 for i in range(self.n + self.u + 2*self.v)]
            for i in range(len(basis)):
                solution[basis[i]] = b[i]
            solution = np.array(solution)
            if(np.all(solution == prev_solution) and cycling == 0):
                print("Cycling detected")
                cycling = 1
            prev_solution = solution
        solution = [0 for i in range(self.n + self.u + 2*self.v)]
        for i in range(len(basis)):
            solution[basis[i]] = b[i]
        solution = np.array(solution)

        print("{:.7f}".format(np.dot(solution, self.c_init)))
        for i in range(self.n):
            print("{:.7f}".format(solution[i]), end = " ")


    def solve(self):
        if(self.v != 0):
            temp = self.phase_1()
            if(temp != None):
                solution, c_hat, basis, A_hat, b = temp
                art_var = np.array([solution[self.n+self.u+2*i] for i in range(self.v)])
                if(np.any(art_var > 1e-16)):
                    print("Infeasible")
                    return
                if(np.dot(solution, c_hat) <= 1e-16):
                    self.phase_2(basis, A_hat, b)
        else:
            basis = np.array([self.n+i for i in range(self.u)])
            self.temp_phase_2(basis, self.A, self.b)
    def state_check(self):
        print("State Check: ")
        print(self.A, self.b)
        print("c_hat", self.c_hat)
        print(self.c)

simp = simplex_method(A, b, c, n_init, u, v)
simp.solve()