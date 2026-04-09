import matplotlib.pyplot as plt

dropout = [20, 30, 40, 50]
time = [15.4, 15.3, 15.2, 15.2]
accuracy = [89.6, 89.4, 89.4, 89.1]

# Plot 1: Time vs Dropout
plt.figure()
plt.plot(dropout, time, marker='o')
plt.xlabel("Dropout Probability (%)")
plt.ylabel("Time (s)")
plt.title("Training Time vs Dropout Probability")
plt.grid(True)
plt.savefig("plots/dropout_time.jpg")
plt.show()

# Plot 2: Accuracy vs Dropout
plt.figure()
plt.plot(dropout, accuracy, marker='o')
plt.xlabel("Dropout Probability (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Dropout Probability")
plt.grid(True)
plt.savefig("plots/dropout_acc.jpg")
plt.show()