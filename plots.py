import matplotlib.pyplot as plt

dropout = [20, 30, 40, 50]
time_single = [15.4, 15.3, 15.2, 15.2]
accuracy_single = [89.6, 89.4, 89.4, 89.1]

time_eight = [123.3, 122.6, 122.7, 122.5]
accuracy_eight = [89.6, 89.5, 89.4, 89.2]

# Plot 1: Time vs Dropout
plt.figure()
plt.plot(dropout, time_single, marker='o')
plt.xlabel("Dropout Probability (%)")
plt.ylabel("Time (s)")
plt.title("Training Time vs Dropout Probability (Single FNN)")
plt.grid(True)
plt.savefig("plots/dropout_single_time.jpg")
plt.show()

# Plot 2: Accuracy vs Dropout
plt.figure()
plt.plot(dropout, accuracy_single, marker='o')
plt.xlabel("Dropout Probability (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Dropout Probability (Single FNN)")
plt.grid(True)
plt.savefig("plots/dropout_single_acc.jpg")
plt.show()

# Plot 3: Time vs Dropout
plt.figure()
plt.plot(dropout, time_eight, marker='o')
plt.xlabel("Dropout Probability (%)")
plt.ylabel("Time (s)")
plt.title("Training Time vs Dropout Probability (Eight FNNs)")
plt.grid(True)
plt.savefig("plots/dropout_eight_time.jpg")
plt.show()

# Plot 4: Accuracy vs Dropout
plt.figure()
plt.plot(dropout, accuracy_eight, marker='o')
plt.xlabel("Dropout Probability (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Dropout Probability (Eight FNNs)")
plt.grid(True)
plt.savefig("plots/dropout_eight_acc.jpg")
plt.show()