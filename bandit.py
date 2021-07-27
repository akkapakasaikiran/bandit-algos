import numpy as np
import matplotlib.pyplot as plt
import argparse

#########################################################################
# Simulating a Multi Armed Bandit Instance
def pull_arm(arm, s, u):
	if np.random.rand() <= true_means[arm]:
		return 1 + s, 1 + u, 1
	else:
		return s, 1 + u, 0


#########################################################################
# Algos

def print_algo_stuff(hz, true_means, num_successes):
	REG = hz*np.max(true_means) - np.sum(num_successes)
	print(ins, algo, rs, ep, hz, REG, sep=', ')

###################################################
# assuming an initial round robin to initialize num_successes
def epsilon_greedy():
	num_successes = np.zeros(n); num_pulls = np.zeros(n)
	for arm in range(n):
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm])
	for t in range(hz-n):
		p_hat = num_successes / num_pulls
		if np.random.rand() <= ep: arm = np.random.randint(n)
		else: arm = np.random.choice(np.flatnonzero(p_hat == np.max(p_hat)))
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm])
		if t+n+1 in hzs and dg: print_algo_stuff(t+n+1, true_means, num_successes)	
	if not dg:
		print_algo_stuff(hz, true_means, num_successes)


###################################################
def ucb():
	num_successes = np.zeros(n); num_pulls = np.zeros(n)
	for arm in range(n):
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm])
	for t in range(hz-n):
		p_hat = num_successes/num_pulls
		ucb_arr = p_hat + (2*np.log(t+n)/num_pulls)**(1/2)
		arm = np.random.choice(np.flatnonzero(ucb_arr == np.max(ucb_arr)))
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm]) 
		if t+n+1 in hzs and dg: print_algo_stuff(t+n+1, true_means, num_successes)
	if not dg:
		print_algo_stuff(hz, true_means, num_successes)


###################################################
small_num = 1e-15

def kl(x, y):
	x = min(max(x, small_num), 1-small_num)
	y = min(max(y, small_num), 1-small_num)
	return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def kl_ucb_val(t, p_hat, u ,c, stopping_val):
	hi = 1-small_num; lo = min(p_hat, hi)
	rhs = (np.log(t) + c*np.log(np.log(t))) / u
	while hi-lo > stopping_val:
		mid = (hi + lo)/2
		if kl(p_hat, mid) <= rhs: lo = mid
		else: hi = mid
	return lo

def kl_ucb():
	num_successes = np.zeros(n); num_pulls = np.zeros(n)
	for arm in range(n):
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm])

	for t in range(hz-n):
		p_hat = num_successes/num_pulls
		kl_ucb_arr = np.array([kl_ucb_val(t+n, p_hat[arm], num_pulls[arm], 0, 1e-5) for arm in range(n)])
		arm = np.random.choice(np.flatnonzero(kl_ucb_arr == np.max(kl_ucb_arr)))
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm]) 
		if t+n+1 in hzs and dg: print_algo_stuff(t+n+1, true_means, num_successes)
	if not dg:
		print_algo_stuff(hz, true_means, num_successes)


###################################################
def thompson():
	num_successes = np.zeros(n); num_pulls = np.zeros(n)
	for t in range(hz):
		samples = np.random.beta(num_successes + 1, num_pulls - num_successes + 1)
		arm = np.random.choice(np.flatnonzero(samples == np.max(samples))) 
		num_successes[arm], num_pulls[arm], _ = pull_arm(arm, num_successes[arm], num_pulls[arm])  
		if t+1 in hzs and dg: print_algo_stuff(t+1, true_means, num_successes)
	if not dg:
		print_algo_stuff(hz, true_means, num_successes)


###################################################
def thompson_hint(jumbled_means):
	num_successes = np.zeros(n); num_pulls = np.zeros(n)
	beliefs = np.full((n,n), 1/n)
	for t in range(hz):
		# samples = [np.random.choice(jumbled_means, p=beliefs[arm]) for arm in range(n)]
		best_mean_probs = beliefs[:,-1]
		arm = np.random.choice(np.flatnonzero(best_mean_probs == np.max(best_mean_probs))) 
		num_successes[arm], num_pulls[arm], val = pull_arm(arm, num_successes[arm], num_pulls[arm])
		beliefs[arm] *= (val * jumbled_means + (1 - val)*(1 - jumbled_means))
		beliefs[arm] /= np.sum(beliefs[arm])
		if t+1 in hzs and dg: print_algo_stuff(t+1, true_means, num_successes)
	if not dg:
		print_algo_stuff(hz, true_means, num_successes)

##########################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--instance', help='Path to the instance file', type=str)
	parser.add_argument('--algorithm', help='''One of epsilon-greedy,
		ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint''', type=str)
	parser.add_argument('--randomSeed', help='A non-negative integer', type=int)
	parser.add_argument('--epsilon', help='A number in [0, 1]', type=float)
	parser.add_argument('--horizon', help='A non-negative integer', type=int)
	parser.add_argument('--dataGen', help='To print intermediate results', action='store_true')
	args = parser.parse_args()

	ins = args.instance
	algo = args.algorithm
	rs = args.randomSeed
	ep = args.epsilon
	hz = args.horizon
	dg = args.dataGen

	if None in [ins, algo, rs, ep, hz, dg]:
		print('usage: python3 bandit.py --instance ins --algorothm algo' + \
			' --randomSeed rs --epsilon eps --horizon hz')
		exit(1)

	hzs = [100, 400, 1600, 6400, 25600, 102400]

	np.random.seed(rs)

	# Reading the bandit instance
	true_means = []
	f = open(ins, 'r')
	for line in f:
		true_means.append(float(line))
	f.close()
	
	n = len(true_means) # number of bandits
	true_means = np.array(true_means)

	if algo == 'epsilon-greedy':
		epsilon_greedy()
	elif algo == 'ucb':
		ucb()
	elif algo == 'kl-ucb':
		kl_ucb()
	elif algo == 'thompson-sampling':
		thompson()
	elif algo == 'thompson-sampling-with-hint':
		thompson_hint(np.sort(true_means))
	else:
		print('Error: Please enter a valid algo')







