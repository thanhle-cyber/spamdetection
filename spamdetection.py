import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import log_loss, confusion_matrix, classification_report
warnings.filterwarnings("ignore")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = text.strip()
    return text

df = pd.read_csv('spam_ham_dataset.csv')
df = df[['text', 'label_num']]
df.columns = ['content', 'label']
df['content'] = df['content'].apply(clean_text)

vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1, solver='saga', warm_start=True)
costs = []

for i in range(50):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_train)
    costs.append(log_loss(y_train, y_prob))

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].plot(costs, color='#1f77b4', lw=2.5)
ax[0].set_title('Cost Function (Log-Loss)', fontsize=14)
ax[0].set_xlabel('Iterations')
ax[0].set_ylabel('Cost')
ax[0].grid(True, alpha=0.3)

z_val = np.linspace(-10, 10, 100)
sigmoid_val = 1 / (1 + np.exp(-z_val))
ax[1].plot(z_val, sigmoid_val, color='#d62728', lw=2.5)
ax[1].axhline(0.5, color='black', ls='--', alpha=0.5)
ax[1].axvline(0, color='black', ls='--', alpha=0.5)
ax[1].set_title('Sigmoid g(z)', fontsize=14)
ax[1].set_xlabel('z = theta^T * x')
ax[1].set_ylabel('Probability h(x)')

plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)
print("\n" + "="*50)
print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT:")
print(classification_report(y_test, y_pred))
print("="*50)
