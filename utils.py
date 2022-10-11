#########################################3
## HELPER FUNCTIONS
#########################################3

#    Dimensions of inputs
#    # W: m x n
#    # P: m x g
#    # U: g x n
#    # L: g x n
#    # x: m x n

TOL = 1e-14

def check_double_stochastic(x, verbose=False):
    mx = 0
    for i in range(x.shape[0]):
        mx = max(mx, sum(x[i,:]) - 1)
        mx = max(mx, 1 - sum(x[i,:]))
        if verbose: print(f'Row {i}: {mx}')
    for j in range(x.shape[1]):
        mx = max(mx, sum(x[:,j]) - 1)
        mx = max(mx, 1 - sum(x[:,j]))
        if verbose: print(f'Row {j}: {mx}')
    if verbose: print(f'Max (double stochastic) violation: {mx}')
    return mx

def check_ranking(x, verbose=False):
    mx = 0
    for i in range(x.shape[0]):
        mx = max(mx, sum(x[i,:]) - 1)
    for j in range(x.shape[1]):
        mx = max(mx, sum(x[:,j]) - 1)
        mx = max(mx, 1 - sum(x[:,j]))
        if verbose: print(f'Position {j}: {max(sum(x[:,j]) - 1, 1 - sum(x[:,j]))}')
    if verbose: print(f'Max (ranking) violation: {mx}')
    return mx

def check_fairness(x, P, L, U, verbose=False):
    m = x.shape[0]
    n = x.shape[1]
    g = P.shape[0]

    o = np.ones((m,1))

    T =  np.zeros((n,n)) # define triangular matrix
    for i in range(n):
        for j in range(i+1):
            T[j][i] = 1

    om = np.ones((m,1))
    on = np.ones((n,1))

    lbVio = (P @ x) @ T - L # lower bound violation
    ubVio = U - (P @ x) @ T # upper bound violation

    done = 0
    mx = 0
    for l in range(g):
        for j, v in enumerate(lbVio[l]):
            if j <= n/3: continue # skip first few positions
            if v < 0:
                if verbose: print(f'LB violated at {j} for group {l} by {-v/(j+1)}')
                mx = max(mx, -v/(j+1)) # fraction violation
                done = 1

    for l in range(g):
        for j, v in enumerate(ubVio[l]):
            if j <= n/3: continue # skip first few positions
            if v < 0:
                if verbose: print(f'UB violated at {j} for group {l} by {-v/(j+1)}')
                mx = max(mx, -v/(j+1)) # fraction violation
                done = 1
    if not done:
        if verbose: print('All const. satisfied')
    return mx

def get_utility(W, x):
    return np.trace(W.T@x)

def complete_ranking(x):
    m = x.shape[0]
    n = x.shape[1]
    sq = np.zeros((m,m))

    for i in range(m):
        for j in range(n):
            sq[i][j] = x[i][j]

    k = n
    b = 1
    for i in range(m):
        s = sum(sq[i,:])
        if k == m:  break
        while s < 1 - TOL:
            tmp = min(1 - s, b)
            sq[i][k] = tmp
            b -= tmp
            s += tmp
            if b < TOL:
                k += 1
                b = 1
                if k == m: break

    check_double_stochastic(sq)
    return sq

def extractBirkhoff(birkhoff, n):
    def extract_first_n_pos(sq, n):
        m = sq.shape[0]
        x = np.zeros((m,n))

        for i in range(m):
            for j in range(n):
                x[i][j] = sq[i][j]
        return x

    a = []
    rankings = []
    for b in birkhoff:
        a += [b[0]]
        rankings += [extract_first_n_pos(b[1], n)]

    return a, rankings

def print_ranking(x):
    m = x.shape[0]
    n = x.shape[1]
    for j in range(n):
        ind = -1
        for i in range(m):
            if x[i][j] > 1 - 1e-5:
                ind = i
                break
        print(f'pos{j}->{i}')



def get_P_equal_grp_sizes(m, n, g):
    P = np.zeros((g,m))
    # Assign equal number of items from each group
    for l in range(g):
        for i in range(l*(m+g-1)//g, min((l+1)*(m+g-1)//g, m)):
            P[l][i] = 1
    
    return P

def get_const_from_dist(dist, m, n, g):
    assert len(dist) == g
    
    U = np.zeros((g,n))
    
    for j in range(n):
        for l in range(g):
            # (j+1+g-1)//g
            U[l][j] = np.ceil((j+1) * dist[l])
            
    return U

def get_upper_const_from_dist_linkedIn_det_greedy(dist, m, n, g):
    assert len(dist) == g
    
    U = np.zeros((g,n))
    
    for j in range(n):
        for l in range(g):
            # (j+1+g-1)//g
            U[l][j] = np.ceil((j+1) * dist[l])
            
    return U

def get_lower_const_from_dist_linkedIn_det_greedy(dist, m, n, g):
    assert len(dist) == g
    
    L = np.zeros((g,n))
    
    for j in range(n):
        for l in range(g):
            # (j+1+g-1)//g
            L[l][j] = np.floor((j+1) * dist[l])
            
    return L

def get_prop_rep_constU(P, m, n, g):
    dist = np.zeros(g)
    
    for l in range(g): dist[l] = np.sum(P[l, :]) / m
    
    return get_const_from_dist(dist, m, n, g)
            
def get_biased_util(P, m, n, g, bias_factor=4):
    w = np.random.rand(m,1)
    r_bit = np.random.random(m)
    
    for i in range(m): 
        #if r_bit[i] <= P[0][i]: w[i] *= bias_factor
        if P[0][i] >= 0.5: w[i] *= bias_factor
        # if P[1][i] >= 0.5: w[i] *= bias_factor
    
    v = [1/np.log(j+1+1) for j in range(n)]
    W = w * v
    return W

def get_true_P(P, m, n, g):
    trueP = copy.deepcopy(P)
    for i in range(m):
        tmp = rng.choice([0,1], p=P[:,i]) 
        trueP[:, i] = np.zeros(g)
        trueP[tmp, i] = 1
    return trueP


def compute_weighted_risk_diff(x, trueP, dist, m, n, g, k=10, verbose=False, P=None):
    rd = 0
    z = 0 # normalizer

    assert (dist > 0).all()

    for j in range(k,n+1,k):
        mx = 0
        for l1, l2 in itertools.product(range(g), repeat=2):
            tmp1 = 0 # count from group 1 in top-j
            tmp2 = 0 # count from group 2 in top-j
            for i in range(m):
                tmp1 += sum(x[i,:j]) * trueP[l1,i]
                tmp2 += sum(x[i,:j]) * trueP[l2,i]
            mx = max(mx, tmp1 / dist[l1] - tmp2 / dist[l2])

            if j >= n-1 and l1==0 and l2==1:
                if verbose: print(f'From G1: {tmp1} and from G2: {tmp2}', flush=True)
        rd += mx / np.log(j+1) * np.min(dist)
        z += (j+1) / np.log(j+1)

    for j in range(n):
        for i in range(m):
            if x[i,j] == 1 and verbose:
                if trueP[0,i] == 1:
                    if P is not None: print(f'G1 ({np.round(P[0, i], 2)})', end=', ')
                    else: print('G1', end=', ')
                else:
                    if P is not None: print(f'G2 ({np.round(P[0, i], 2)})', end=', ')
                    else: print('G2', end=', ')
    if verbose: print('.')
    rd = 1 - rd / z

    return rd

def compute_weighted_selec_lift(x, trueP, dist, m, n, g, k=10, verbose=False, P=None):
    rd = 0
    z = 0 # normalizer

    assert (dist > 0).all()
    assert sum(dist) > 1 - 1e-5

    for j in range(k,n+1,k):
        mx = 100
        for l1, l2 in itertools.product(range(g), repeat=2):
            tmp1 = 0 # count from group 1 in top-j
            tmp2 = 0 # count from group 2 in top-j
            for i in range(m):
                tmp1 += sum(x[i,:j]) * trueP[l1,i]
                tmp2 += sum(x[i,:j]) * trueP[l2,i]
            mx = min(mx, tmp1/(tmp2+1e-18) * dist[l2] / dist[l1])
        rd += mx / np.log(j+1)
        z += 1 / np.log(j+1)

    rd = rd / z

    return rd

def compute_weighted_KL_div(x, trueP, dist, m, n, g, k=10, verbose=False, P=None):
    def kl(p, q):
        assert len(p) == len(q)
        return np.sum(np.where(p*q != 0, p * np.log(p / q), 0))


    rd = 0
    z = 0 # normalizer

    # ensure the target distribution represents all groups (otherwise, KL-divergence can be ∞)
    assert (dist > 0).all()

    for j in range(k,n+1,k):
        mx = 0
        cnt = np.zeros(g)
        for l in range(g):
            for i in range(m):
                cnt[l] += sum(x[i,:j]) * trueP[l,i]

        cnt /= sum(cnt)

        rd += kl(cnt, dist) / np.log(j+1)
        z +=  - np.log(np.min(dist)) / np.log(j+1)

    rd = 1 - rd / z

    return rd


def print_prefix_counts(x, l, m, n, g):
    for j in range(n):
        cnt = 0
        for i in range(m):
            cnt += sum(x[i,:j]) * trueP[l,i]
        if False: print(f'{j}->{cnt}')
    return cnt



