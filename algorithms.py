#########################################################
## IMPLEMENTATIONS OF RANKING ALGORITHMS 
#########################################################


##################################
## LP APPROACH SINGH-JOACHIMS-KDD18
##################################
# 1. Solve the LP-relaxation of the fair ranking problem (with imputed protected attributes); 
# 2. Use birkhoff_von_neumann_decomposition to round the fractional solution
# 3. Instead of exposure this controls for LU fairness
def noisy_rank_basic_rounding(W, PT, L, U, getBirkhoff=True, verbose=False):
    # Solves the noisy fair LP then uses birkhoff_von_neumann_decomposition
    m = W.shape[0]
    n = W.shape[1]
    g = PT.shape[0]
    
    x = cp.Variable((m,n), boolean=False)
    o = np.ones((m,1))
    
    T =  np.zeros((n,n)) # define triangular matrix 
    for i in range(n):
        for j in range(i+1):
            T[j][i] = 1
            
    om = np.ones((m,1))
    on = np.ones((n,1))
    
    const = []
    # fairness constraints 
    const += [(PT @ x) @ T <= U, (PT @ x) @ T >= L]
    # at most one position per item
    const += [ x@on <= om]
    # exactly one item per position
    const += [ om.T@x == on.T]
    # ranking between 0 and 1
    const += [ x <= 1, x >= 0]
    
    prob = cp.Problem(cp.Maximize(cp.trace(W.T@x)), const);
    
    try:
        st = time.time()
        prob.solve(solver=cp.GUROBI, parallel=True, verbose=False)
        if verbose: print(f'Time taken for LP: {time.time() - st}')
        # cplex_params={"timelimit": 30}
        # solver=cp.CPLEX,
        if type(x)==type(None) or type(x.value) == type(None):
            print('here')
            prob.solve(verbose=True)        
    
    except Exception as e:
        print(e)
        print("Could not solve fair_select")
        return -1
    
    x = x.value 
    check_ranking(x, verbose=False)
    check_fairness(x, PT, L, U, verbose=False)
    
    if verbose: print(f'Utility: {get_utility(W, x)}')
    
    if not getBirkhoff: return x, 0 
    
    st = time.time()
    sq = complete_ranking(x)
    result = birkhoff_von_neumann_decomposition(sq)
    if verbose: print(f'Time taken for decomposition: {time.time() - st}')
    
    return x, result


##################################
## UTILITY MAXIMIZING RANKING
##################################
def unconstrained_ranking(W, PT, getBirkhoff=True, verbose=False):
    # Solves the noisy fair LP then uses birkhoff_von_neumann_decomposition
    m = W.shape[0]
    n = W.shape[1]
    g = PT.shape[0]
        
    
    # ensure that each item has exactly one protected attribute
    for i in range(m): assert sum(PT[:, i]) == 1
    
    st = time.time()
    # iterator for each group
    g_it = np.zeros(g, dtype=int) 
    # list of id's of items in each of the groups
    g_id = [[] for l in range(g)] 
    # list of utilities of items in each group
    g_W = [[] for l in range(g)] 
    # indices of items in each group sorted in decreasing order of utility
    g_argsort = [[] for l in range(g)]

    for i in range(m):
        for l in range(g):
            if PT[l][i] == 1: 
                g_id[l].append(i) # add id to list
                g_W[l].append(W[i][0]) # add utility to list
    
    # set sorted indices of the utility
    for l in range(g): g_argsort[l] = np.argsort(-np.array(g_W[l]))
    
    # ranking
    x = np.zeros((m,n))
    
    for j in range(n): # iterate over positions
        mx = -1
        mx_grp = -1 # store the best group with a slack fairness constraint
        
        cur_ind = np.zeros(g, dtype=int) # index of the best unranked candidate in each group
        
        for l in range(g):
            if g_it[l] >= len(g_id[l]): continue # cannot add candidate from group l
            
            cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l
            
            if g_W[l][cur_ind[l]] > mx:
                mx = g_W[l][cur_ind[l]]
                mx_grp = l # chose group l
        
        # ensure problem is feasible
        assert mx_grp != -1
        
        mx_id = g_id[mx_grp][cur_ind[mx_grp]] # id of the best unranked candidate from the chosen group
        x[mx_id, j] = 1 # place candidate at position j
        g_it[mx_grp] += 1 # record that we ranked anoter candidate from group mx_grp
        
    if verbose: print(f'Time taken for Greedy: {time.time() - st}')
    return x



##################################
## GREEDY CELIS-STRASZAK-VISHNO-ICALP-2018
##################################
def greedy_fair_ranking(W, PT, L, U, getBirkhoff=True, verbose=False):
    # Solves the noisy fair LP then uses birkhoff_von_neumann_decomposition
    m = W.shape[0]
    n = W.shape[1]
    g = PT.shape[0]

    # ensure that each item has exactly one protected attribute
    for i in range(m): assert sum(PT[:, i]) == 1

    # ensure there are no lower bounds
    assert np.sum(np.abs(L)) == 0


    st = time.time()
    # iterator for each group
    g_it = np.zeros(g, dtype=int)
    # list of id's of items in each of the groups
    g_id = [[] for l in range(g)]
    # list of utilities of items in each group
    g_W = [[] for l in range(g)]
    # indices of items in each group sorted in decreasing order of utility
    g_argsort = [[] for l in range(g)]

    for i in range(m):
        for l in range(g):
            if PT[l][i] == 1:
                g_id[l].append(i) # add id to list
                g_W[l].append(W[i][0]) # add utility to list

    # set sorted indices of the utility
    for l in range(g): g_argsort[l] = np.argsort(-np.array(g_W[l]))

    # ranking
    x = np.zeros((m,n))

    for j in range(n): # iterate over positions
        mx = -1
        mx_grp = -1 # store the best group with a slack fairness constraint

        cur_ind = np.zeros(g, dtype=int) # index of the best unranked candidate in each group

        for l in range(g):
            if g_it[l] >= U[l,j] or g_it[l] >= len(g_id[l]): continue # cannot add candidate from group l

            cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

            if g_W[l][cur_ind[l]] > mx:
                mx = g_W[l][cur_ind[l]]
                mx_grp = l # chose group l

        if mx_grp == -1:
            # allow the ranking to violate the fairness constraints
            # (by selecting the itiem with the highest utility from the remaining items)
            for l in range(g):
                if g_it[l] >= len(g_id[l]): continue # cannot add candidate from group l

                cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

                if g_W[l][cur_ind[l]] > mx:
                    mx = g_W[l][cur_ind[l]]
                    mx_grp = l # chose group l

        # ensure problem is feasible
        assert mx_grp != -1

        mx_id = g_id[mx_grp][cur_ind[mx_grp]] # id of the best unranked candidate from the chosen group
        x[mx_id, j] = 1 # place candidate at position j
        g_it[mx_grp] += 1 # record that we ranked anoter candidate from group mx_grp

    if verbose: print(f'Time taken for Greedy: {time.time() - st}')
    return x


##################################
## OUR ALGO: LP+ROUNDING
##################################
# 1. Linear programming
# 2. Rounding by [Chekuri-Vondrak-Zenklusen-SODA-2011]
def helper_merge_using_cvz(I, J, a, b, p):
    # {(u, v - m) for u, v in M.items() if u < m}

    def order(I):
        assert type(I) == type(set())
        tmp = list(I)
        for i in range(len(tmp)):
            if tmp[i][0] > tmp[i][1]:
                tmp[i] = (tmp[i][1], tmp[i][0])
        tmp = set(tmp)
        assert type(tmp) == type(set())
        return tmp

    def get_dict(I):
        assert type(I) == type(set())
        tmp = {}
        for i in I: tmp[i] = 0
        assert type(tmp) == type(dict())
        return tmp

    def get_set(I):
        assert type(I) == type(dict())

        tmp = set()
        for i in I: tmp.add(i)

        assert type(tmp) == type(set())
        return tmp

    I = copy.deepcopy(I)
    J = copy.deepcopy(J)

    I = order(I)
    J = order(J)

    def get_cc(edges):
        assert type(edges) == type(set())


        # todo
        g = {} # adjacency list
        v = set()
        t = 1 # "time"


        def dfs(u, v_t, t):
            v_t[u] = t
            for v in g[u]:
                if v_t[v] != 0: continue
                dfs(v, v_t, t)


        # build graph
        for e in edges:
            if e[0] not in g: g[e[0]] = []
            if e[1] not in g: g[e[1]] = []
            g[e[0]].append(e[1])
            g[e[1]].append(e[0])
            v.update({e[0],e[1]})

        v_t = {vv: 0 for vv in v} # time for each vertex

        for vv in v:
            if v_t[vv] == 0:
                dfs(vv, v_t, t)
                t += 1

        cc = [set() for i in range(t-1)]

        for e in edges: cc[v_t[e[0]] - 1].add(e)

        # returns a list of sets
        for comp in cc:
            assert type(comp) == type(set())
            for e in comp:
                assert type(e) == type(tuple())

        return cc

    def is_path(edges):
        assert type(edges) == type(set())

        # checks if the set of edges is a path or a cycle

        fg = True

        g = {} # adjacency list
        v = set() # vertex list

        def dfs(u, p, v_t):
            v_t[u] = 1

            for v in g[u]:
                if v == p: continue
                # found a back edge
                if v_t[v] != 0:  return False
                if not dfs(v, u, v_t): return False

            return True

        # build graph
        for e in edges:
            if e[0] not in g: g[e[0]] = []
            if e[1] not in g: g[e[1]] = []
            g[e[0]].append(e[1])
            g[e[1]].append(e[0])
            v.update({e[0],e[1]})

        v_t = {vv: 0 for vv in v} # time for each vertex

        for vv in v:
            fg = dfs(vv, -1, v_t)
            break # only need to run dfs for one iteration

        for vv in v:
            if v_t[vv] == 0:
                assert False # Input edges should be a connected path or cycle

        return fg

    def get_ordered_edges(edges):
        # todo
        # traverse edges to get an ordered path

        ordered_edges = []

        g = {} # adjacency list
        v = set() # vertex list

        def dfs(u, p, v_t):
            v_t[u] = 1
            for v in g[u]:
                if v == p: continue

                if v_t[v] != 2: ordered_edges.append((u, v))

                # Only run dfs on non-visited vertices
                if v_t[v] == 0: dfs(v, u, v_t)
            v_t[u] = 2

        # build graph
        for e in edges:
            if e[0] not in g: g[e[0]] = []
            if e[1] not in g: g[e[1]] = []
            g[e[0]].append(e[1])
            g[e[1]].append(e[0])
            v.update({e[0],e[1]})

        v_t = {vv: 0 for vv in v} # time for each vertex

        st = None

        # Set st to a degree 1 vertex if it exists
        for vv in v:
            if len(g[vv]) == 1:
                st = vv
                break

        # if no degree 1 vertex, set st to the first vertex in v
        if st is None:
            for vv in v:
                st = vv
                break

        dfs(st, -1, v_t)

        return ordered_edges

    ##############################################

    def get_paths_int(edges, I, J, p):
        assert type(edges) == type(set())
        assert type(I) == type(set())
        assert type(J) == type(set())
        # done
        paths = []

        if len(edges) <= 2 * p:
            for i in range(p):
                paths.append(copy.deepcopy(edges))

            tmp = edges & J

            for i, e in enumerate(tmp): paths[i].difference_update({e})
        elif is_path(edges): # edges is a path
            # traverse edges to get an ordered path
            ordered_edges = get_ordered_edges(edges)

            fg = 0
            tmp = ordered_edges[0]
            if tmp not in I and (tmp[1], tmp[0]) not in I: fg = 1

            # for each j repeat the construction
            for j in range(1, p+1):
                # starting from the 2j-th edge remove every 2p-th edge
                # Note: index starts from 0

                to_delete = set() # this set record the edges to be deleted
                it = 2 * j - 1 - fg # if the first edges is in J, start removing from 1-st edge

                # fill to_delete
                while it < len(ordered_edges):
                    to_delete.add(ordered_edges[it])
                    it += 2 * p

                assert len(to_delete & I) == 0

                # delete edges
                edges_copy = copy.deepcopy(edges)
                edges_copy.difference_update(to_delete)

                paths.extend(get_cc(edges_copy))
        else: # edges is a cycle
            # traverse edges to get an ordered path
            ordered_edges = get_ordered_edges(edges)

            N = len(ordered_edges)

            for i in range(N):
                tmp = ordered_edges[i]
                if tmp not in I and (tmp[1], tmp[0]) not in I: continue

                path_int = set()

                for j in range(2*p-1):
                    path_int.add(ordered_edges[(i + j) % N])

                paths.append(path_int)

        assert type(paths) == type(list())

        for path in paths:
            if type(path) != type(set()):
                assert False
            for e in path:
                if type(e) != type(tuple()):
                    assert False

        return paths

    def get_paths(IJ, I, J, p):
        # done
        paths = []

        connected_components = get_cc(IJ)

        for cc in connected_components:
            paths_int = get_paths_int(cc, I, J, p)
            paths.extend(paths_int)

        for i in range(len(paths)): paths[i] = order(paths[i])

        return paths

    while I != J:
        # done
        # IJ is IΔJ
        IJ = I^J

        # compute paths
        paths_1 = get_paths(IJ, I, J, p)
        paths_2 = get_paths(IJ, J, I, p)

        if len(paths_1) == 0 or len(paths_2) == 0:
            assert False

        # set parameters
        rho = (p-1) / len(paths_1)
        sigma = p / len(paths_2)
        rho_1 = np.ones(len(paths_1)) / len(paths_1)
        rho_2 = np.ones(len(paths_2)) / len(paths_2)

        # set probabilities
        f = b * sigma / (a*rho + b*sigma)
        prob_1 = f * rho_1
        prob_2 = (1 - f) * rho_2
        prob = np.concatenate([prob_1, prob_2])
        prob /= sum(prob)

        ind = rng.choice(len(prob), size = 1, p = prob)[0]


        if ind < len(paths_1):
            I.symmetric_difference_update(paths_1[ind])
            for e in I:
                if type(e) != type((1,2)):
                    eval(debug('e'))
                    eval(debug('ind'))
                    eval(debug('paths_1[ind]'))
                    eval(debug('I'))
        else:
            ind -= len(paths_1)
            J.symmetric_difference_update(paths_2[ind])
            for e in J:
                if type(e) != type((1,2)):
                    eval(debug('e'))
                    eval(debug('ind'))
                    eval(debug('paths_2[ind]'))
                    eval(debug('J'))

    return I

def noisy_rank_cvz_rounding(W, PT, L, UU, verbose=False):
    # Solves the noisy fair LP then uses rounding by [ChekuriVondrakZenklusen-SODA11]

    U = copy.deepcopy(UU)

    m = W.shape[0]
    n = W.shape[1]
    g = PT.shape[0]

    x = cp.Variable((m,n), boolean=False)
    o = np.ones((m,1))

    T =  np.zeros((n,n)) # define triangular matrix
    for i in range(n):
        for j in range(i+1):
            T[j][i] = 1

    om = np.ones((m,1))
    on = np.ones((n,1))

    const = []
    for i in range(len(U)): U[i] *= 1 + 1.0/20*np.sqrt(1/(i+1))

        
    # fairness constraints
    const += [(PT @ x) @ T <= U, (PT @ x) @ T >= L]
    # at most one position per item
    const += [ x@on <= om]
    # exactly one item per position
    const += [ om.T@x == on.T]
    # ranking between 0 and 1
    const += [ x <= 1, x >= 0]

    prob = cp.Problem(cp.Maximize(cp.trace(W.T@x)), const);

    try:
        st = time.time()
        prob.solve(solver = cp.GUROBI, parallel=True, verbose=False)
        if verbose: print(f'Time taken for LP: {time.time() - st}')
        if type(x)==type(None) or type(x.value) == type(None):
            print('Failed to solve. Trying again with verbose. CVZ rounding.')
            prob.solve(solver=cp.GUROBI, parallel=True, verbose=True)
    except Exception as e:
        print(e)
        print("Could not solve fair_select")
        return -1

    if verbose: print('Found ranking with LP', flush=True)
    x = x.value

    if verbose: print(f'Utility: {get_utility(W, x)}')

    decomposition = fast_decomposition(x)
    if verbose: print(f'Time taken by fast-decomposition: {time.time() - st}')

    p_tot, _, m_tot = decomposition[0]
    precision = 100

    for i in range(len(decomposition)):
        match = decomposition[i][2]
        for e in match:
            if e[0] >= m or e[1] < m:
                assert False

    if verbose: print(f'Length of decomposition: {len(decomposition)}')

    st = time.time()
    if verbose:
        for (p, r, match) in decomposition: print(f'sum(r): {np.sum(r)}')
    if verbose:
        for (p, r, match) in decomposition: print(f'p: {p}')

    for (p, r, match) in decomposition[1:]:
        m_tot = helper_merge_using_cvz(match, m_tot, p, p_tot, precision)
        p_tot += p
    if verbose: print(f'Time taken by merge: {time.time() - st}')

    x_cvz = np.zeros((m, n))
    for e in m_tot:
        x_cvz[e[0]][e[1] - m] = 1

    return x_cvz

##################################
## SUBSET SELECT ALGO [MEHROTRA-CELIS-FAccT-2021]
##################################
def subset_selection_algorithm(w, PT, l, u, n, verbose=False):
    # Outputs the ranking subset in expectation over the randomness in the candidates
    m = w.shape[0]
    p = PT.shape[0]
    u = np.array([u]).T

    x = cp.Variable((m,1))

    o = np.ones((m,1))
    om = np.ones((m,1))
    on = np.ones((n,1))

    prob = cp.Problem( cp.Maximize(w.T * x),[PT@x <= u, o.T@x == n,x <= 1,x >= 0]);
    try:
        prob.solve(solver = cp.GUROBI, parallel=True, verbose=False)
        if type(x)==type(None) or type(x.value) == type(None):
            print('Failed to solve. Trying again with verbose. Subsetselection-based.')
            prob.solve(solver=cp.GUROBI, parallel=True, verbose=True)
    except Exception as e:
        print(e)
        print("Could not solve fair_select")
        return -1
    x = list(x)
    x = np.array([np.clip(y.value[0],0,1) for y in x])
    x = rand_round_solution(x,m,n)

    li = []
    for i in range(m):
        if x[i] == 1:
            li += [(-w[i], i)]
    li.sort()

    rank = np.zeros((m,n))
    for j in range(n): rank[li[j][1], j] = 1

    return rank

##################################
## DET-GREEDY [GEIYK-ET.-AL]
##################################
# ASSUMPTION: W is a rank 1 metric
def linkedIn_det_greedy(W, PT, L, U, getBirkhoff=True, verbose=False):
    # Assumption: W is a rank 1 metric
    m = W.shape[0]
    n = W.shape[1]
    g = PT.shape[0]

    # ensure that each item has exactly one protected attribute
    for i in range(m): assert sum(PT[:, i]) == 1

    # assert np.sum(np.abs(L)) == 0


    st = time.time()
    # iterator for each group
    g_it = np.zeros(g, dtype=int)
    # list of id's of items in each of the groups
    g_id = [[] for l in range(g)]
    # list of utilities of items in each group
    g_W = [[] for l in range(g)]
    # indices of items in each group sorted in decreasing order of utility
    g_argsort = [[] for l in range(g)]

    for i in range(m):
        for l in range(g):
            if PT[l][i] == 1:
                g_id[l].append(i) # add id to list
                g_W[l].append(W[i][0]) # add utility to list

    # set sorted indices of the utility
    for l in range(g): g_argsort[l] = np.argsort(-np.array(g_W[l]))

    # ranking
    x = np.zeros((m,n))

    for j in range(n): # iterate over positions
        mx = -1
        mx_grp = -1 # store the best group with a slack fairness constraint

        cur_ind = np.zeros(g, dtype=int) # index of the best unranked candidate in each group

        below_min = []
        below_max = []

        for l in range(g):
            if g_it[l] >= len(g_id[l]): continue
            if g_it[l] < L[l,j]: below_min.append(l)

        for l in range(g):
            if g_it[l] >= len(g_id[l]): continue
            if g_it[l] < U[l,j]: below_max.append(l)

        for l in below_min:
            cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

            if g_W[l][cur_ind[l]] > mx:
                mx = g_W[l][cur_ind[l]]
                mx_grp = l # chose group l

        if mx_grp == -1:
            for l in below_max:
                cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

                if g_W[l][cur_ind[l]] > mx:
                    mx = g_W[l][cur_ind[l]]
                    mx_grp = l # chose group l

        if mx_grp == -1:
            # allow the ranking to violate the fairness constraints
            # (by selecting the itiem with the highest utility from the remaining items)
            for l in range(g):
                if g_it[l] >= len(g_id[l]): continue # cannot add candidate from group l

                cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

                if g_W[l][cur_ind[l]] > mx:
                    mx = g_W[l][cur_ind[l]]
                    mx_grp = l # chose group l

        # ensure problem is feasible
        assert mx_grp != -1

        mx_id = g_id[mx_grp][cur_ind[mx_grp]] # id of the best unranked candidate from the chosen group
        x[mx_id, j] = 1 # place candidate at position j
        g_it[mx_grp] += 1 # record that we ranked anoter candidate from group mx_grp

    if verbose: print(f'Time taken for det_greedy: {time.time() - st}')
    return x

