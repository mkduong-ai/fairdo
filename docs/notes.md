# Code
- stop criterion for fairness optimizer [x]
- allow other hyperparameters []
- fix only contains one class []
- individual fairness []


## Ideas
- [x] CD takes a metric to optimize on
  - [ ] window size
- [] Adds data points for which the metric is improved

## Todo
- [x] Implement new methods
- improve main.py
	- [x] Move metrics outside
	- [x] use the metrics outside of main.py
	- [x] plan to move other things outside of main.py

# Paper

## Approach Contribution
- Metrics can be chosen beforehand
  - used fairness metrics for datasets
- Works for any group metric and individual metric notion
  - iFair: individual metric notion satisfies statistical parity and equalised odds empirically
- Does not depend on the classifier (or regression task)
- creates pareto optima almost in every case wrt classification metric and fairness notion
- Requires one hyperparameter

## Evaluation
- [ ] 2-3 datasets (compas, adult, german)
- [ ] clf metric
  - F1, Balanced Accuracy, AUC
- [ ] Fairness Notion
  - AVG Odds, MI, statistical parity
- [ ] Classifier
  - DT, LR
