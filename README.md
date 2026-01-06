# Chess App

This is a standard Chess-App built using PySide6 Library for the Front End and standard MinMax search with Alpha-Beta Pruning Optimizations.

# Search Algorithm 

The Chess App uses the standard MinMax search with Alpha-Beta Pruning for search optimizations. It uses Quiescence Search beyond MinMax to mitigate the Horizon Effect.

The search Algorithm first uses Iterative Deepening for lower depths to find prespective move. It then uses Lazy Symmetric Multiprocessing to parallel the search the across multi threads to speed up searches for later depths. 

The board evaluation algorithm is hand crafted.