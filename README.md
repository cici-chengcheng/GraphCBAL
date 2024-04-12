# Class-Balanced Active Learning for Graph Neural Networks via Reinforcement Learning
This is the code of the paper GraphCBAL: Class-Balanced Active Learning for Graph Neural Networks via Reinforcement Learning

## Dependencies
matplotlib==2.2.3
networkx==2.4
scikit-learn==0.22.1
numpy==1.16.3
scipy==1.2.1
torch==1.7.0

## Data
We have provided Cora, Pubmed, Citeseer, Reddit whose data format have been processed and can be directly consumed by our code. 

## Train

Use ```train.py``` to train the active learning policy on one cora dataset. 

We train ```GraphCBAL``` using the labeled training graph of ```cora``` with query budgets of ```35```, and we want to save the trained model in ```GraphCBAL.pkl```, then use the following commend: 

```
python -m src.train --datasets cora --budgets 35  --save 1  --savename GraphCBAL  
```

We train ```GraphCBAL++``` using the labeled training graph of ```cora``` with query budgets of ```35```, and we want to save the trained model in ```GraphCBAL++.pkl```, then use the following commend: 

```
python -m src.train --datasets cora --budgets 35  --use_major_class 1 --punishment 0.05  --save 1  --savename GraphCBAL++  
```

where ```use_major_class``` and ```punishment``` refer to the Majority-Score state feature and the penalty score in the punishment mechanism.

Please refer to the source code to see how to set the other arguments. 

## Test
Use ```test.py``` to test the learned active learning policy on unlabeled test graphs. For example, we have an unlabeled test graph ```Citeseer``` with a query budget of ```120```, and if we want to test the policy ```GraphCBAL.pkl```, then use the following commend: 

```
python -m src.test  --datasets citeseet --budgets 120  --modelname GraphCBAL
```

If we want to test the policy ```GraphCBAL++.pkl```, then use the following commend:

```
python -m src.test  --datasets citeseet --budgets 120  --use_major_class 1 --modelname GraphCBAL++
```

Please refer to the source code to see how to set the other arguments. 


