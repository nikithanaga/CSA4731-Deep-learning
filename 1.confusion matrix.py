import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Actual and predicted labels
actual = np.array(['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted = np.array(['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

# Unique labels
labels = ['Dog', 'Not Dog']

# Initialize a confusion matrix with zeros
cm = np.zeros((2, 2), dtype=int)

# Populate the confusion matrix
for a, p in zip(actual, predicted):
    i = labels.index(a)
    j = labels.index(p)
    cm[i, j] += 1

# Plot
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
