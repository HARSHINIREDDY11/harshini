import os
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib.pyplot as plt

# Performance data
models = ['CNN Baseline', 'Swin Single-View', 'EcoView (Swin + 3D CNN)']
accuracy = [85.2, 91.4, 94.8]
precision = [84.1, 90.8, 94.2]
recall = [85.2, 91.4, 94.8]
f1_score = [84.6, 91.1, 94.5]

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
data = [accuracy, precision, recall, f1_score]

# Create a grouped bar chart
x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each metric
for i, (metric, values) in enumerate(zip(metrics, data)):
    ax.bar(x + i * width, values, width, label=metric)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Percentage (%)')
ax.set_title('Performance Comparison: EcoView vs Baselines')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models)
ax.legend()

# Save the plot
plt.tight_layout()
plt.savefig('plots/performance_graph.png')
print("Performance graph saved to plots/performance_graph.png")
