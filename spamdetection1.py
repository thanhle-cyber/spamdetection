import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

# ====================== GRADIENT DESCENT ======================
def gradient_descent(X, y, theta, alpha=0.15, iterations=1200):
    m = len(y)
    cost_history = []
    
    print("Đang train Logistic Regression bằng Gradient Descent...")
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        if i % 300 == 0 or i == iterations - 1:
            print(f"Iteration {i+1}/{iterations} - Cost = {cost:.6f}")
    
    return theta, cost_history

# ====================== VẼ HÀM SIGMOID ======================
def plot_sigmoid():
    z = np.linspace(-10, 10, 200)
    sig = sigmoid(z)
    plt.figure(figsize=(8, 5))
    plt.plot(z, sig, color='blue', linewidth=2, label='Hàm Sigmoid σ(z)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Đồ thị hàm Sigmoid (Logistic Function)', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('σ(z) = 1 / (1 + e^(-z))')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ====================== CHƯƠNG TRÌNH CHÍNH ======================
print("=== SPAM DETECTION - LOGISTIC REGRESSION TỪ SCRATCH ===")

# 1. Đọc dữ liệu
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')

print(f"Dataset shape: {df.shape}")
print("Các cột:", list(df.columns))
print("\nPhân bố label ban đầu:")
print(df['label'].value_counts())

# 2. Xử lý label: chuyển 'ham' → 0, 'spam' → 1
# Dùng cột 'label_num' nếu có (nhiều file spam_ham_dataset.csv có sẵn cột này)
if 'label_num' in df.columns:
    y = df['label_num'].values.astype(float)
    print("Sử dụng cột 'label_num' (đã là 0/1)")
else:
    # Chuyển từ string 'ham'/'spam'
    df['label'] = df['label'].str.lower().str.strip()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    y = df['label_num'].values.astype(float)
    print("Đã chuyển label từ string sang số (0=ham, 1=spam)")

# 3. Vector hóa text bằng TF-IDF
print("\nĐang vector hóa văn bản bằng TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000,      # Giới hạn để train nhanh hơn
    min_df=2,
    max_df=0.95
)

X_vec = vectorizer.fit_transform(df['text']).toarray()
print(f"Feature matrix sau vector hóa: {X_vec.shape}")

# 4. Shuffle và chia đúng 4000 train - 1000 test
np.random.seed(42)
indices = np.random.permutation(len(df))
X_vec = X_vec[indices]
y = y[indices]

X_train = X_vec[:4000]
X_test  = X_vec[4000:5000]
y_train = y[:4000]
y_test  = y[4000:5000]

print(f"\nTrain set: {X_train.shape[0]} emails")
print(f"Test set : {X_test.shape[0]} emails")

# Thêm bias term (cột toàn 1)
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias  = np.c_[np.ones(X_test.shape[0]), X_test]

# Khởi tạo theta = 0
theta = np.zeros(X_train_bias.shape[1])

# 5. Train model
theta_final, cost_history = gradient_descent(X_train_bias, y_train, theta, alpha=0.15, iterations=1200)

# 6. Vẽ Cost Function
plt.figure(figsize=(9, 5))
plt.plot(cost_history, color='darkblue', linewidth=2)
plt.title('Cost Function J(θ) qua các iterations', fontsize=14, fontweight='bold')
plt.xlabel('Iterations')
plt.ylabel('Cost J(θ)')
plt.grid(True, alpha=0.3)
plt.show()

# 7. Vẽ hàm Sigmoid
plot_sigmoid()

# 8. Dự đoán và đánh giá trên 1000 emails test
print("\n=== BÁO CÁO PHÂN LOẠI CHI TIẾT TRÊN TEST SET ===")
h_test = sigmoid(np.dot(X_test_bias, theta_final))
y_pred = (h_test >= 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Tập Test (1000 emails)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

acc = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác tổng thể: {acc*100:.2f}%")

print("\n=== HOÀN THÀNH CHƯƠNG TRÌNH ===")