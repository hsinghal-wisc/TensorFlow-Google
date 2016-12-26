Matrix Multiplication

python bigmatrixmultiplication.py (simple to understand code, taking around 4 minutes)

Formula used

tr(A2)=∑i∑jtr(AijAji)
Optimisation - Distribute the trace evenly across the cluster.

Synchronous SGD

python synchronoussgd.py

Learning rate : 0.1
Time per iteration in sync mode : ~3 sec

Implementation details:

a.Used gather operation since the model vector was huge and sparse. Gather operation ,when used at master vm,
can lead to 100 times optimisation due to reduction in NW transfers
b.Used scatter add to optimize the addition of a dense tensor 'w' and the gradient received as a sparse tensor.


Asynchronous SGD

./launch_asyncsgd.sh (launches asyncsgd.py)

Learning rate : 0.1
Time per iteration on each worker in async mode : ~1 sec (per update from each worker)

Implementation details:

All the decision taken in Sync mode applies here. In addition we did:
All the session were executed in vm-1 task-0

The above implementations can be optimised and generalised further (for eg. batch processing)
