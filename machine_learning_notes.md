# 📘 Machine Learning Notes

## 🔎 What is Machine Learning?
Machine Learning (ML) is a subset of **Artificial Intelligence (AI)** that allows systems to **learn from data** and improve performance on tasks **without being explicitly programmed**.


- Instead of writing rules, we provide **data + algorithms** → the model finds patterns and makes predictions.
- Example: Predicting house prices, spam detection, recommendation systems

---

## 🧠 Types of Machine Learning

### 1️⃣ Supervised Learning
- **Definition**: Model learns from a labeled dataset (input → output).
- **Goal**: Predict outputs for new/unseen inputs.
- **Examples**:
  - **Regression** → Predict continuous values (e.g., house prices).
  - **Classification** → Predict categories (e.g., spam vs not spam).
- **Algorithms**: Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, KNN.

---

### 2️⃣ Unsupervised Learning
- **Definition**: Model learns from **unlabeled data** (no output labels).
- **Goal**: Find hidden structure or patterns in data.
- **Examples**:
  - **Clustering** → Grouping customers by behavior.
  - **Dimensionality Reduction** → Reducing features while keeping important info (PCA).
- **Algorithms**: KMeans, Hierarchical Clustering, PCA.

---

### 3️⃣ Reinforcement Learning (RL)
- **Definition**: Model (agent) learns by interacting with an environment, receiving **rewards or penalties**.
- **Goal**: Maximize cumulative reward through trial and error.
- **Examples**: Self-driving cars, game-playing AI (Chess, Go).
- **Key Terms**: Agent, Environment, Reward, Policy, Value Function.

---

### 4️⃣ Semi-Supervised Learning
- **Definition**: Uses both **labeled + unlabeled data** to improve learning.
- **Example**: A small set of medical images labeled by doctors, combined with many unlabeled images.

---

### 5️⃣ Online Learning
- **Definition**: Model learns **incrementally** as new data comes in (instead of retraining from scratch).
- **Useful for**: Streaming data, stock price prediction, fraud detection.

---

## ⚙️ Key Concepts in ML
- **Features** → Input variables (e.g., sepal length, petal width).  
- **Labels/Targets** → Output to predict (e.g., Iris species).  
- **Training Data** → Data used to teach the model.  
- **Testing Data** → Data used to evaluate performance.  
- **Overfitting** → Model performs well on training data but poorly on new data.  
- **Underfitting** → Model is too simple, fails to capture patterns.  
- **Cross-Validation** → Technique to evaluate model performance reliably. 

---

## 📊 ML Workflow
1. Collect Data  
2. Preprocess Data (cleaning, scaling, encoding)  
3. Split into Training & Testing sets  
4. Train the Model  
5. Evaluate Model (accuracy, precision, recall, etc.)  
6. Deploy / Use the Model   

---

## 📏 Evaluation Metrics in Machine Learning

### 🔹 For Regression
- **Mean Absolute Error (MAE)** → Average absolute difference between predictions & actual values.  
- **Mean Squared Error (MSE)** → Penalizes large errors more.  
- **Root Mean Squared Error (RMSE)** → Square root of MSE, same units as target.  
- **R² Score (Coefficient of Determination)** → How well the model explains variance (1 = perfect, 0 = poor).  

### 🔹 For Classification
- **Accuracy** → Correct predictions / Total predictions.  
- **Precision** → Out of predicted positives, how many are actually positive?  
- **Recall (Sensitivity/TPR)** → Out of actual positives, how many were correctly predicted?  
- **F1 Score** → Harmonic mean of Precision & Recall (useful for imbalanced data).  
- **Confusion Matrix** → Table showing True Positives, False Positives, False Negatives, True Negatives.  
- **ROC Curve & AUC** → Tradeoff between sensitivity & specificity.  

### 🔹 For Clustering
- **Silhouette Score** → How well clusters are separated and compact.  
- **Inertia (SSE)** → Sum of squared distances within clusters (used in KMeans Elbow Method).  
- **Adjusted Rand Index (ARI)** → Measures similarity between predicted clusters and ground truth.  

---

## 🛠️ Popular ML Libraries in Python
- **NumPy** → Numerical computations  
- **Pandas** → Data manipulation  
- **Matplotlib / Seaborn** → Visualization  
- **Scikit-Learn** → ML algorithms & preprocessing  
- **TensorFlow / PyTorch** → Deep learning  

---

✨ These notes cover **ML concepts, types, evaluation metrics, and workflow** , serving a quick reference for both study and projects.
