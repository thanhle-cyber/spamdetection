import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ====================== CÔNG THỨC TOÁN HỌC - NEWTON'S METHOD ======================
"""
Logistic Regression sử dụng Newton's Method (Pure Newton-Raphson)

1. Cost function (Binary Cross Entropy):
   J(θ) = - (1/m) Σ [ y⁽ⁱ⁾ log(h⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-h⁽ⁱ⁾) ]
   với h = σ(Xθ) = 1 / (1 + exp(-Xθ))

2. Gradient:
   ∇J(θ) = (1/m) Xᵀ (h - y)

3. Hessian matrix:
   H(θ) = (1/m) Xᵀ diag(h ⊙ (1-h)) X

4. Newton's update rule:
   θ_new = θ_old - H⁻¹ * ∇J(θ)

Ưu điểm: Hội tụ cực nhanh, thường chỉ cần 5-15 iterations.
"""

# ====================== HÀM SIGMOID ======================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ====================== COST FUNCTION ======================
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5
    cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# ====================== NEWTON'S METHOD ======================
def newton_method(X, y, theta, max_iter=20, tol=1e-8):
    """
    Train Logistic Regression bằng Newton's Method
    Trả về: theta_final, cost_history
    """
    m, n = X.shape
    cost_history = []
    
    print("Dang train bang NEWTON'S METHOD (max 20 iterations)...")
    
    for i in range(max_iter):
        h = sigmoid(np.dot(X, theta))
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        # Gradient
        grad = (1 / m) * np.dot(X.T, (h - y))
        
        # Hessian = (1/m) * X.T @ diag(h*(1-h)) @ X
        s = h * (1 - h)                                   # vector
        Hessian = (1 / m) * np.dot(X.T, (s[:, np.newaxis] * X))
        
        # Newton's update: theta = theta - inv(H) * grad
        # Sử dụng solve để ổn định hơn inv
        try:
            delta = np.linalg.solve(Hessian, grad)
            theta = theta - delta
        except np.linalg.LinAlgError:
            print("Hessian không khả nghịch tại iteration", i+1)
            break
        
        # In tiến trình
        print(f"Iteration {i+1:2d}/{max_iter} - Cost = {cost:.6f}")
        
        # Dừng sớm nếu hội tụ
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Newton's method da hoi tu som tai iteration {i+1}!")
            break
    
    return theta, cost_history

# ====================== VẼ HÀM SIGMOID ======================
def plot_sigmoid():
    z = np.linspace(-10, 10, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(z, sigmoid(z), color='blue', linewidth=2, label='Hàm Sigmoid σ(z)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Đồ thị hàm Sigmoid', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('σ(z) = 1 / (1 + e^(-z))')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ====================== CHƯƠNG TRÌNH CHÍNH ======================
print("=== SPAM DETECTION - LOGISTIC REGRESSION (NEWTON'S METHOD) ===")

# 1. Đọc dữ liệu
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')

# Xử lý label
if 'label_num' in df.columns:
    y = df['label_num'].values.astype(float)
else:
    df['label'] = df['label'].str.lower().str.strip()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    y = df['label_num'].values.astype(float)

# 2. Vector hóa TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, min_df=2, max_df=0.95)
X_vec = vectorizer.fit_transform(df['text']).toarray()

# 3. Chia 4000 train - 1000 test
np.random.seed(42)
indices = np.random.permutation(len(df))
X_vec = X_vec[indices]
y = y[indices]

X_train = X_vec[:4000]
X_test  = X_vec[4000:5000]
y_train = y[:4000]
y_test  = y[4000:5000]

# Thêm bias term
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias  = np.c_[np.ones(X_test.shape[0]), X_test]

# Khởi tạo theta = 0
theta = np.zeros(X_train_bias.shape[1])

# 4. Train bằng Newton's Method
theta_final, cost_history = newton_method(X_train_bias, y_train, theta, max_iter=20)

# ====================== KẾT QUẢ TRẢ VỀ (theo yêu cầu) ======================
print("\n" + "="*70)
print("KẾT QUẢ TRẢ VỀ CHO NHÓM:")

score = accuracy_score(y_test, (sigmoid(np.dot(X_test_bias, theta_final)) >= 0.5).astype(int))
print(f"Score (Accuracy)          = {score*100:.4f}%")

print(f"Cost history (len = {len(cost_history)}):")
print([round(c, 6) for c in cost_history])

intercept = theta_final[0]
coef = theta_final[1:]
print(f"Intercept (bias)          = {intercept:.6f}")
print(f"Coefficients shape        = {coef.shape}")

print("="*70)

# 5. Vẽ Cost Function
plt.figure(figsize=(9, 5))
plt.plot(cost_history, color='red', linewidth=2, marker='o')
plt.title("Cost Function J(θ) - Newton's Method", fontsize=14, fontweight='bold')
plt.xlabel('Iterations')
plt.ylabel('Cost J(θ)')
plt.grid(True, alpha=0.3)
plt.show()

# 6. Vẽ Sigmoid
plot_sigmoid()

# 7. Báo cáo phân loại
print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT (Test set 1000 emails):")
h_test = sigmoid(np.dot(X_test_bias, theta_final))
y_pred = (h_test >= 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Newton\'s Method')
plt.show()

print("\nHOAN THANH! Da su dung pure Newton's Method thay cho LBFGS.")