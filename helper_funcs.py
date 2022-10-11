#####################################################
## LOGISTICAL/STORING FUNCTIONS
#####################################################

debug = lambda str : f"print(\"{str}\",\"=\",eval(\"{str}\"))"

rcParams.update({
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'figure.figsize': (10,6),
})

def file_str():
    """ Auto-generates file name."""
    now = datetime.datetime.now()
    return now.strftime("H%HM%MS%S_%m-%d-%y")

rand_string = lambda length: ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def pdf_savefig():
    """ Saves figures as pdf """
    fname = file_str()+rand_string(5)
    plt.savefig(home_folder+f"/figs/{fname}.pdf")

def eps_savefig():
    """ Saves figure as encapsulated postscript file (vector format)
        so that it isn't pixelated when we put it into a pdf. """
    pdf_savefig()

def handler(signum, frame):
	print("Forever is over!")
	raise Exception("end of time")

def handler2(signum, frame):
	print("Forever is over!")
	raise Exception("end of time")

def handler3(signum, frame):
	print("Forever is over!")
	raise Exception("end of time")

def rand_round_solution(x,m,n):
    rx=np.array([0]*m);
    ind=rng.choice([i for i in range(len(x))], n, replace=False, p=x/np.sum(x))
    # for i in range(n): rx[li[i][1]]=1
    for i in ind: rx[i]=1
    rx=rx.reshape((len(rx),1))
    return np.array(rx)

def get_fairness(num_picked, n, fair_metric):
    p = len(num_picked)
    selec_lft = False; r_diff = False; custom = False
    if fair_metric == 'selec_lft': selec_lft = True
    if fair_metric == 'r_diff': r_diff = True
    if fair_metric == 'custom': custom = True
    if store_val: return val
    elif selec_lft:
        mi = 2
        for i,j in itertools.product(range(p),range(p)):
            mi = min(mi, num_picked[i]/num_picked[j] if num_picked[j] else 0.0)
        return mi
    elif r_diff:
        mi = -1
        for i,j in itertools.product(range(p),range(p)):
            mi = max(mi, abs(num_picked[i]-num_picked[j])/n)
        return 1-mi
    elif custom:
        mi = 2
        for i,j in itertools.product(range(p),range(p)):
            mi = min(mi, p/(p-1.0) * (1 - num_picked[i]/n))
        return mi
    else: print("Must choose a fairness definition!"); raise NotImplementedError

def get_rng(rand_gen=None):
    if rand_gen is not None: return rand_gen
    else: return rng


