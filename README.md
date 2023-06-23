# Uncertainty Estimation with CIFAR10 using Monte Carlo Inference

<div style="display:flex;justify-content:center;">
    <img src="https://github.com/jmayank23/UncertaintyEstimation_CIFAR10/assets/27727185/520ac562-d370-4013-b6e5-c0ae930705e7" alt="ship" width="330"/>
    <img src="https://github.com/jmayank23/UncertaintyEstimation_CIFAR10/assets/27727185/fc4f789f-226b-4feb-bbcb-d8b638292c87" alt="uncertainty" width="430"/>
</div>



## Introduction
This code trains a convolutional neural network (CNN) on the CIFAR-10 dataset and then implements Monte Carlo Dropout Inference to estimate model uncertainty. The documentation will cover the importance of model uncertainty, an explanation of Monte Carlo Dropout Inference, and the sequence of steps involved in the code.

## Importance of Model Uncertainty
In deep learning, model uncertainty refers to the confidence or uncertainty associated with the predictions made by a model. It is crucial to understand model uncertainty for several reasons:
- **Reliability assessment:** Model uncertainty helps assess the reliability of predictions and identify cases where the model's predictions may be less trustworthy.
- **Safety-critical applications:** In safety-critical domains such as autonomous driving or medical diagnosis, knowing the model's uncertainty is essential for making informed decisions and taking appropriate actions.
- **Error detection and human intervention:** Model uncertainty can indicate cases where the model encounters inputs significantly different from the training data, helping identify potential errors and prompting human intervention or further examination.

## Monte Carlo Dropout Inference
Monte Carlo Dropout Inference is a technique used to estimate uncertainty in deep learning models, particularly when dropout regularization is applied. Dropout is a regularization technique that randomly sets a fraction of neurons to zero during training, preventing the network from relying too heavily on specific neurons and promoting robust representations.

Monte Carlo Dropout Inference extends dropout beyond training and applies it during the inference or testing phase. It involves running multiple forward passes with dropout enabled to obtain multiple predictions for each input. By sampling different sets of dropped-out neurons in each forward pass, Monte Carlo Dropout Inference captures the effect of dropout and enables uncertainty estimation.

The uncertainty estimation process involves aggregating the predictions obtained from multiple forward passes. The aggregated predictions provide a measure of uncertainty or confidence in the model's predictions. Metrics such as variance or entropy can be calculated from the aggregated predictions to quantify the uncertainty.

## Sequence of Steps in the Code
The code can be divided into the following steps:

1. **Data Loading and Preparation:** The CIFAR-10 dataset is loaded and preprocessed. It is divided into training and validation sets, and data loaders are created for efficient batch processing.

2. **Neural Network Architecture:** The neural network architecture is defined using the `Net` class, which consists of convolutional and fully connected layers. Dropout is applied to the second fully connected layer.

3. **Model Training:** The neural network is trained using a training loop. The model is optimized using the Adam optimizer and the Cross-Entropy Loss function. Training metrics (loss and accuracy) are calculated for each epoch.

4. **Validation:** The trained model is evaluated on the validation set to assess its performance. Validation metrics (loss and accuracy) are calculated.

5. **Loss and Accuracy Visualization:** The training and validation loss and accuracy curves are plotted to visualize the model's learning progress.

6. **Monte Carlo Dropout Inference:** The `monte_carlo_dropout_inference` function is defined to perform Monte Carlo Dropout Inference. It takes the trained model, validation dataloader, the number of Monte Carlo samples, and the number of classes as inputs. It returns predictions obtained from multiple forward passes.

7. **Uncertainty Estimation:** Monte Carlo Dropout Inference is applied to the validation set, and predictive mean and variance are calculated for each batch of samples.

8. **Visualization of Uncertainty:** An example image from the validation set is randomly selected, and the uncertainty estimates for each class are displayed. A bar plot is generated to visualize the predicted probabilities with uncertainty.

The code provides a practical implementation of Monte Carlo Dropout Inference for uncertainty estimation in deep learning models and demonstrates the visualization of uncertainty using the CIFAR-10 dataset.
