# HumanActionDetection
This project applies deep learning with TensorFlow/Keras to detect and classify human actions from video frames. It includes data preprocessing, feature extraction, CNN/LSTM model training, and evaluation, enabling real-time action recognition for smart applications.

Mobile Health Raw Dataset -> https://www.kaggle.com/datasets/gaurav2022/mobile-health

Accuracy Obtained for 'k' values ->

<img width="559" height="700" alt="Screenshot 2025-08-26 at 9 31 14 AM" src="https://github.com/user-attachments/assets/f705bf28-aff5-49c4-8604-212bb1b27e89" />


🚀 Features --->

1.) Dataset: mHealth (wearable sensor signals for daily activities)
2.)EDA: distributions, correlations, trend exploration (Matplotlib/Seaborn)
3.)Preprocessing:
4.)Label encoding of activities
5.)Handling imbalance via resampling
6.)RobustScaler for outlier-resistant scaling
7.)Feature work: optional Lasso for feature selection/regularization

Models compared (scikit-learn):
a.)Logistic Regression
b.)SVM
c.)Random Forest
d.)K-Nearest Neighbors
e.)Gaussian Naive Bayes
f.)Decision Tree
g.)Evaluation metrics: Accuracy, Precision, Recall, F1; confusion matrix visualization.
h.)Reproducible pipeline runnable end-to-end in Jupyter/Colab (includes Drive mounting code)

📈 Results --->

Example model runs reported:
Accuracy ~93.92%, Precision ~93.74%, Recall ~93.64%, F1 ~93.33%
Additional runs around 88.73% and 87.04% accuracy (with matching precision/recall/F1)
Confusion matrix plots are generated to inspect per-class performance.
In your runs, the top model achieved ~93.9% accuracy with balanced precision/recall, indicating strong activity classification quality on mHealth.
