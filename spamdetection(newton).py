import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# HAM SIGMOID
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# HAM COST
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5
    cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# NEWTON METHOD
def newton_method(X, y, theta, max_iter=15, lambda_reg=0.5):
    m, n = X.shape
    cost_history = []
    
    print("Training Newton's Method with max_iter = " + str(max_iter))
    
    for i in range(max_iter):
        h = sigmoid(np.dot(X, theta))
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        grad = (1 / m) * np.dot(X.T, (h - y)) + (lambda_reg / m) * theta
        grad[0] = (1 / m) * np.sum(h - y)
        
        s = h * (1 - h)
        Hessian = (1 / m) * np.dot(X.T, s[:, np.newaxis] * X) + (lambda_reg / m) * np.eye(n)
        Hessian[0, 0] -= (lambda_reg / m)
        
        try:
            delta = np.linalg.solve(Hessian, grad)
            theta = theta - delta
        except np.linalg.LinAlgError:
            print("Hessian singular at iteration " + str(i+1))
            break
        
        print("Iteration " + str(i+1) + "/" + str(max_iter) + " - Cost = " + str(round(cost, 6)))
    
    print("Finished! Cost history has " + str(len(cost_history)) + " values.")
    return theta, cost_history

# MAIN PROGRAM
print("=== SPAM DETECTION - NEWTON METHOD (Optimized) ===")

df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')

if 'label_num' in df.columns:
    y = df['label_num'].values.astype(float)
else:
    df['label_num'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})
    y = df['label_num'].values.astype(float)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 1),
    min_df=3,
    max_df=0.95
)

X_vec = vectorizer.fit_transform(df['text']).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=1000, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], "| Features:", X_train.shape[1])

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias  = np.c_[np.ones(X_test.shape[0]), X_test]

theta = np.zeros(X_train_bias.shape[1])

theta_final, cost_history = newton_method(X_train_bias, y_train, theta, max_iter=15, lambda_reg=0.5)

# KET QUA TRA VE
score = accuracy_score(y_test, (sigmoid(np.dot(X_test_bias, theta_final)) >= 0.5).astype(int))

print("\n" + "="*60)
print("SCORE (Accuracy) =", round(score*100, 4), "%")
print("Cost history has", len(cost_history), "values")
print("Intercept =", round(theta_final[0], 6))
print("Coef shape =", theta_final[1:].shape)
print("="*60)

# VE COST GRAPH
plt.figure(figsize=(10, 6))
plt.plot(cost_history, color='red', linewidth=2, marker='o')
plt.title("Cost Function - Newton's Method")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

# BAO CAO
h_test = sigmoid(np.dot(X_test_bias, theta_final))
y_pred = (h_test >= 0.5).astype(int)
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], digits=4))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("\nDONE!")