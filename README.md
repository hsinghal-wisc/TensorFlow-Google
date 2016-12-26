
2 Parts:


Multiplicaton of large matrices
=======
Logistic Regression - Machine Learning using Synchronous and Asynchronous Stochastic Gradient Descent
=======


Matrix Multiplication
=======
Find the trace (sum of diagonal elements) of square of a random n×n matrix, A, where n=100,000
Unlike Spark, TF does not provide a big-data abstraction. So we have to break down the large matrix (or generate
them in pieces), and schedule computations on the smaller pieces to achieve our end-goal. We can represent a
large matrix as:

A=[A11A12…A1n;A21A22…A2n;…;An1…Ann]

The trace of A2 can now be computed as
∑i∑jtr(AijAji)

Run file - python bigmatrixmultiplication.py (simple to understand code, running time around 4 minutes)
Optimisation - Distribute the trace evenly across the cluster.


Logistic Regression
=======
Implemented Binary Logistic Regression (LR) to learn a model for predicting if a user will click on
advertisement or not. 


(i)Synchronous SGD

Run - python synchronoussgd.py

Learning rate : 0.1
Time per iteration in sync mode : ~3 sec

Implementation details:

a.Used gather operation since the model vector was huge and sparse. Gather operation ,when used at master vm,
can lead to 100 times optimisation due to reduction in NW transfers
b.Used scatter add to optimize the addition of a dense tensor 'w' and the gradient received as a sparse tensor.


(ii)Asynchronous SGD
Asynchronous SGD

./launch_asyncsgd.sh (launches asyncsgd.py)

Learning rate : 0.1
Time per iteration on each worker in async mode : ~1 sec (per update from each worker)

Implementation details:

All the decision taken in Sync mode applies here. In addition we did:
All the session were executed in vm-1 task-0

(iii)The above implementations can be optimised and generalised further (for eg. batch processing)
