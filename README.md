# bandit-algos

A simple python implementation of a few algorithms which sample arms of a multi-armed bandit with the goal of minimizing regret. 

## Usage

` python3 bandit.py --instance ins --algorothm algo --randomSeed rs --epsilon eps --horizon hz `
where
- `ins` is the path to a bandit instance file (examples given in `instances/`)
- `aglo` is one of `epsilon-greedy`, `ucb`, `kl-ucb`, `thompson-sampling`, and `thompson-sampling-with-hint`
- `rs` is a non-negative integer used to make the results deterministic
- `ep` is a number in \[0, 1\]
- `hz` is a non-negative integer

The program simulates the multi-armed bandit encoded in `ins`, runs `algo` for `hz` time steps, and generates the cumulative regret `reg`. 
The output is in the following format. 

`ins, algo, rs, eps, hz, reg`  

This was a course assignment of [CS 747](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs747-a2020/index.html): Foundations of Intelligent and Learning Agents
done in my third year as a UG at IITB. The assignment's problem statement can be found [here](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs747-a2020/pa-1/programming-assignment-1.html). 