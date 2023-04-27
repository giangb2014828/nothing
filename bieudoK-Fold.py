import matplotlib.pyplot as plt
x = ['KNN', 'Bayes', 'DecisionTree']
y1 = [0.54, 0.43, 0.17]
y2 = [0.28, 0.14, 0]
y3 = [0.7, 0.83, 0.04]

plt.bar(x, y1, width=0.2, align='edge')
plt.bar([i + 0.2 for i in range(len(x))], y2, width=0.2, align='edge')
plt.bar([i + 0.4 for i in range(len(x))], y3, width=0.2, align='edge')

for i, v in enumerate(y1):
    plt.annotate(str(v), xy=(i - 0.11, v + 0.01))
for i, v in enumerate(y2):
    plt.annotate(str(v), xy=(i + 0.1, v + 0.01))
for i, v in enumerate(y3):
    plt.annotate(str(v), xy=(i + 0.3, v + 0.01))

plt.title('Biểu đồ đánh giá các mô hình')
plt.legend(['Precision Score', 'Recall Score', 'F1 Score'])
plt.ylim(0, 1.2)
plt.show()