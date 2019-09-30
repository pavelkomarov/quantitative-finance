import cvxopt as cvx
import numpy as np

# minimize: (1/2)*x'*H*x + f'*x
# subject to: A*x <= b
#			Aeq*x = beq
#https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
#https://gist.github.com/garydoranjr/1878742
def quadprog(H, f, A, b, Aeq=None, Beq=None):
	cvx.solvers.options['show_progress'] = False
	P, q, G, h, A, b = convert_to_cvxopt(H, f, A, b, Aeq, Beq)#outputs correspond to inputs
	results = cvx.solvers.qp(P, q, G, h, A, b)
	return (np.array(results['x']), results['status']=='optimal')

#Convert everything to cvxopt-style matrices 
def convert_to_cvxopt(H, f, A, b, Aeq, beq):
	P = cvx.matrix(H, tc='d')
	q = cvx.matrix(f, tc='d')
	G = cvx.matrix(A, tc='d')
	h = cvx.matrix(b, tc='d')
	if Aeq is None: A = None
	else: A = cvx.matrix(Aeq, tc='d')
	if beq is None: b = None
	else: b = cvx.matrix(beq, tc='d')

	return P, q, G, h, A, b
	
# run a little test to make sure the result is the same as matlab
if __name__ == "__main__":
	H = np.array([[1,2],[2,15]])
	f = np.array([[0],[1]])
	A = np.array([[1,1],[1,3]])
	b = np.array([[3],[-5]])

	bestx = quadprog(H, f, A, b)
	print(bestx)
