# Machine Learning Course Projects

A comprehensive collection of machine learning implementations covering fundamental to advanced topics in data analysis, visualization, classical ML, deep learning, and generative models.

---

## Requirements

All required Python dependencies are listed in requirements.txt.

1️⃣ Create and activate a virtual environment (recommended)

`python -m venv venv`
`source venv/bin/activate`        # Linux / macOS
`venv\Scripts\activate`           # Windows

(If using Conda, activate your environment instead.)

---

2️⃣ Install dependencies (CPU version)

`pip install -r requirements.txt`
This installs all libraries including PyTorch (CPU-only).

---
3️⃣ (Optional) GPU support for PyTorch (CUDA)

If you have an NVIDIA GPU, install PyTorch with CUDA support before installing the remaining dependencies.

Visit the official PyTorch selector:
`https://pytorch.org/get-started/locally/`

Example (CUDA 12.1):

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
`pip install -r requirements.txt`

**Make sure your CUDA version matches your installed NVIDIA drivers.**

---

## 01 - Dataset Analysis Visualization

- Generate and analyze 10,000 synthetic student records with attributes (gender, major, program, GPA)
- Create comprehensive visualizations including distributions, conditional analyses, and pairplots
- Implement and compare simple vs stratified sampling techniques
- Perform specialized sampling strategies: gender-balanced, GPA-uniform, and program-major balanced cohorts

---

## 02 - K-Nearest Neighbors

- Build a KNN classifier from scratch to predict student gender based on features
- Implement custom `PerFeatureTransformer` class for handling categorical and numerical features
- Analyze performance across different k values (1-21) and distance metrics (Euclidean, Manhattan, Cosine)
- Evaluate single-feature vs multi-feature predictions using F1-scores and accuracy metrics

---

## 03 - Linear Regression

- Implement polynomial regression with L1 and L2 regularization for GPA prediction
- Compare models across polynomial degrees 1-6 with varying regularization strengths
- Identify optimal hyperparameters using validation set performance
- Analyze feature importance and interpret non-zero weights from regularized models

---

## 04 - K-Means Clustering

- Implement K-Means clustering algorithm from scratch with `fit()`, `predict()`, and `getCost()` methods
- Determine optimal number of clusters using Elbow Method (WCSS) and Silhouette Score
- Evaluate cluster quality and interpret clustering results
- Compare custom implementation with sklearn's built-in K-Means

---

## 05 - Gaussian Mixture Models

- Build GMM class from scratch implementing the Expectation-Maximization (EM) algorithm
- Implement `fit()`, `getMembership()`, and `getLikelihood()` methods
- Determine optimal clusters using BIC (Bayesian Information Criterion) and Silhouette Method
- Visualize likelihood convergence across iterations

---

## 06 - Image Segmentation

- Perform color-based image segmentation on satellite images using custom GMM implementation
- Segment images into three distinct regions (Land, Water, Vegetation) using k=3 Gaussian components
- Create dynamic visualization videos showing EM algorithm convergence frame-by-frame
- Display original image, segmented output, and log-likelihood plots side-by-side in video format

---

## 07 - Principle Component Analysis

- Implement PCA from scratch with `fit()`, `transform()`, and `checkPCA()` methods
- Apply dimensionality reduction to MNIST dataset (28×28 pixels) reducing to 500, 300, 150, and 30 dimensions
- Analyze explained variance vs number of principal components
- Visualize reconstruction quality by projecting reduced dimensions back to original space

---

## 08 - PCA Classification

- Investigate effect of PCA dimensionality reduction on KNN classifier performance
- Apply custom PCA to 8×8 MNIST digits dataset with varying component counts [2, 5, 10, 20, 30, 40, 50, 64]
- Train KNN classifiers with different k values [5, 25, 50, 100] on reduced datasets
- Analyze relationship between principal components, k values, and classification accuracy

---

## 09 - Data Transformation Clustering

- Analyze modified MNIST dataset (1,000 altered images) to identify data transformations
- Design and justify appropriate transformation techniques to address observed differences
- Compare clustering performance before and after transformation using evaluation metrics
- Explain why transformations improved clustering outcomes

---

## 10 - Neural Network From Scratch

- Implement MLP from scratch including Linear layers, activation functions (ReLU, Tanh, Sigmoid, Identity)
- Build complete training pipeline with forward/backward passes, gradient accumulation, and early stopping
- Model complex Belgium-Netherlands border (Baarle-Nassau) using coordinate-based prediction
- Achieve 91% accuracy through architecture optimization and hyperparameter tuning

---

## 11 - Feature Mappings For Image Reconstruction

- Compare three feature representations: Raw pixels, Taylor series expansion, and Fourier feature mapping
- Reconstruct grayscale (smiley.png) and RGB (cat.jpg) images using coordinate-based MLPs
- Analyze convergence speed and reconstruction quality across different feature mappings
- Evaluate performance on blurred images with Gaussian blur levels σ = 0 to 10

---

## 12 - Autoencoders

- Build deep autoencoder from scratch for MNIST image reconstruction using custom MLP components
- Apply autoencoder to anomaly detection on Labeled Faces in the Wild (LFW) dataset
- Train exclusively on "George W Bush" images as normal class, detect other faces as anomalies
- Analyze effect of bottleneck dimensions on anomaly detection performance using ROC/PR curves

---

## 13 - Multi-Task CNN FMNIST

- Implement multi-task CNN jointly performing classification and regression on Fashion-MNIST
- Classification: predict 10 clothing categories; Regression: predict normalized pixel intensity ("ink")
- Optimize joint loss: L = λ₁L_CE + λ₂L_MSE with varying weight combinations
- Visualize intermediate feature maps to interpret learned representations at different layers

---

## 14 - Image Colorization CNN

- Build encoder-decoder CNN for CIFAR-10 image colorization (32×32 RGB images)
- Treat colorization as classification: map pixels to 24 representative color clusters
- Use learned upsampling via ConvTranspose2d layers in decoder
- Perform extensive hyperparameter tuning (learning rate, batch size, filters, kernels) using wandb sweeps

---

## 15 - Decision Tree From Scratch

- Implement decision tree classifier from scratch with custom node classes (Leaf and Internal)
- Build functions for best binary split selection and recursive tree training
- Apply to Amazon product review sentiment classification (bag-of-words, 7729 features)
- Visualize tree structure and analyze decision boundaries

---

## 16 - Decision Tree Classification

- Train decision trees on text sentiment classification using Scikit-Learn
- Perform grid search over max_depth and min_samples_leaf hyperparameters
- Visualize trained trees using ASCII representation and analyze node structures
- Optimize using balanced accuracy metric with proper train/validation splits

---

## 17 - Random Forest Classification

- Build ensemble Random Forest classifier for Amazon review sentiment analysis
- Analyze feature importance to identify most predictive vocabulary terms
- Tune max_features, max_depth, and n_estimators through grid search
- Compare performance: Simple DT vs Best DT vs Simple RF vs Best RF

---

## 18 - KDE Foreground Detection

- Implement Kernel Density Estimation (KDE) class from scratch using only numpy
- Support multiple kernel types (Gaussian, Triangular, Uniform) with configurable bandwidth
- Perform foreground detection by fitting KDE on background image and classifying test pixels
- Optimize bandwidth and kernel selection for best segmentation results

---

## 19 - Time Series RNN

- Discover discrete-time recurrence from univariate sequences using RNNs
- Build predictors mapping history vectors to next values: x̂ₖ = g(xₖ₋₁, xₖ₋₂, ..., xₖ₋ₚ)
- Identify closed-form analytical recurrence from learned RNN weights
- Evaluate single-step and autoregressive multi-step predictions using MAE/MSE

---

## 20 - Time Series Forecasting

- Forecast cumulative GitHub star growth trajectories for multiple repositories
- Compare classical (ARMA) vs deep learning (RNN, 1D CNN) forecasting methods
- Implement proper temporal splits ensuring no data leakage from future
- Evaluate single-timestep and increasing-window multi-step forecasts across horizons

---

## 21 - Variational Autoencoders

- Implement VAE with encoder, reparameterization trick, and decoder for Fashion-MNIST
- Optimize combined loss: reconstruction (BCE) + KL divergence to standard normal prior
- Investigate β-VAE: vary weighting between reconstruction and KL terms
- Analyze effect of frozen latent parameters (µ, σ) on generation quality and diversity

---

## Repository Structure

Each notebook is self-contained with:
- Problem description and objectives
- Implementation from scratch (where applicable)
- Experiments and hyperparameter tuning
- Visualization and analysis
- Results and conclusions

---