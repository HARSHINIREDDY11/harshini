import matplotlib.pyplot as plt
import numpy as np

# Performance data
models = ['CNN Baseline', 'Swin Single-View', 'EcoView (Swin + 3D CNN)']
accuracy = [85.2, 91.4, 94.8]
precision = [84.1, 90.8, 94.2]
recall = [85.2, 91.4, 94.8]
f1_score = [84.6, 91.1, 94.5]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#1f77b4')
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#ff7f0e')
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#2ca02c')
rects4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#d62728')

ax.set_ylabel('Percentage (%)')
ax.set_title('Performance Comparison: EcoView vs Baselines')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
