import matplotlib.pyplot as plt
x = ['KNN', 'Bayes', 'DecisionTree']
y1 = [0.5, 0.61, 0.34]
y2 = [0.72, 0.61, 0.41]
y3 = [0.58, 0.27, 0.09]

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