# bandit-algos

A simple python implementation of a few algorithms which sample arms of a multi-armed bandit with the goal of minimizing regret. 

### Usage

` $ python3 bandit.py --instance ins --algorothm algo --randomSeed rs --epsilon eps --horizon hz `

Where
- `ins` is the path to a bandit instance file (a few examples given in `instances/`)
- `aglo` is one of `epsilon-greedy`, `ucb`, `kl-ucb`, `thompson-sampling`, or `thompson-sampling-with-hint`
- `rs` is a non-negative integer used to make the results deterministic
- `ep` is a number in \[0, 1\], a parameter for the epsilon-greedy algorithm
- `hz` is a non-negative integer

The program simulates the multi-armed bandit encoded in `ins`, runs `algo` for `hz` time steps, and calculates the cumulative regret `reg`. 
The output format is `ins, algo, rs, eps, hz, reg`. 

This project was done for a course assignment of IITB's [CS 747](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs747-a2020/index.html): Foundations of Intelligent and Learning Agents. The assignment's problem statement can be found [here](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs747-a2020/pa-1/programming-assignment-1.html). 
