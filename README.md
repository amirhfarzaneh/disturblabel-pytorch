# DisturbLabel-PyTorch
PyTorch implementation of [DisturbLabel: Regularizing CNN on the Loss Layer](https://arxiv.org/abs/1605.00055) [CVPR 2016]

## DisturbLabel introduction:
The authors propose adding label noise deliberately to improve the generalization of image classification algorithms. They claim it is the first attempt at regularizing the network at **loss layer**, unlike other methods:

Based on Multinoulli distribution, the DisturbLabel algorithm changes a small portion of the mini-batch labels regardless of the ground-truth labels. The noise paramater `alpha` controls the amount of noise.

## Implementation details:

- platform details:
```
python = 3.7
pytorch = 1.0.1
gpu = NVIDIA GTX 1080Ti
```

- training parameters:
  - **dataset** = MNIST
  - **network** = modified LeNet (as in the paper)
  - **bacth size** = 64 (*not mentioned in the paper*)
  - **epochs** = 100 (as in the paper)
  - **learning rate**: scheduled as [10e-3, 10e-4, 10e-5, 10e-6] changing at epochs [40, 60, 80] (as in the paper)
  - **optimization**: SGD with momentum = 0.9 (*not mentioned in the paper*)

## Running the code:
- to run the code with **no regularization**:
```
python main.py --mode=noreg
```
- to run the code with **dropout**:
```
python main.py --mode=dropout
```
- to run the code with **disturblabel**:
```
python main.py --mode=disturblabel --alpha=10  # --alpha can be any number (10, 20, 40 in the paper)
```
- to run the code with **no regularization**:
```
python main.py --mode=both --alpha=10  # --alpha can be any number (10, 20, 40 in the paper)
```

## Results:

- MNIST test error evaluating DisturbLabel with `alpha=20` w/o data augmentation

| method       | test error | paper |
|--------------|------------|-------|
| no Reg.      | 0.77       | 0.86  |
| Dropout      | 0.59       | 0.68  |
| DisturbLabel | 0.59       | 0.66  |
| both Reg.    | 0.51       | 0.63  |

## Conclusion:

In general I'm getting better results than what the paper is reporting; but the trend is similar. DisturbLabel has a regularization effect and improves the model. Similar to the paper results, Dropout and DisturbLabel individually improve results on a same scale but their combination performs the best.

### TO DO:
- [ ] test on CIFAR10 dataset
- [ ] report the results and insert their corresponding plots for different noise `alpha` values.