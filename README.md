## Fairness Regularization RecSys (on-progress)

### Technology stack
* Python 3.5
* Numpy
* Pandas
* Scipy
* Matplotlib
* Jupyter
* Django (UI)

### Pre-processing
* Removing long-tail data, user & item who has < 30 rating
* Divide item popularity, short-head (top 20%) and medium-tail

### Parameters
* Ranking objective function.
* ILBU item distance d(i, j) == True if item is on the same set
* lambda reg = 0.000006 - 0.000009 (positive)
* Weighting Importance ALS disabled or = False
* K-latent factor == 50 (ALS for Personalized Ranking, Takacs, Tikk)
* Iteration == 30 (ALS for Personalized Ranking, Takacs, Tikk)

### Performance Test
* NDCG@10 (quality rank)
* APT
* Medium-tail coverage

### How to Build
* install package in requirements.txt
* install static npm package for ui
* select model & dataset in config.py