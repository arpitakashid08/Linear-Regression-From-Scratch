import numpy as np
import matplotlib.pyplot as plt
x= np.array([1,2,3,4], dtype=float)
y= np.array([3,5,7,9], dtype=float)
w=0.0
b=0.0
learning_rate=0.01
epochs=1000
for epoch in range(epochs):
    y_pred = w*x + b
    loss = np.mean((y_pred-y)**2)
    dw=2*np.mean((y_pred-y)*x)
    db=2*np.mean(y_pred-y)
    w=w-learning_rate*dw
    b=b-learning_rate*db
    if epoch%100==0:
        print(f"Epoch {epoch} | Loss:{loss: .4f} | w: {w: .4f} | b: {b: .4f}")

        print("\nTraining completed!")
        print(f"Final weight(w): {w: .4f}")
        print(f"Final weight(w): {w: .4f}")
        print(f"Final bias (b): {b: .4f}")
        x_test = 5
        y_test_pred = w*x_test + b
        print(f"Prediction for x = {x_test}: {y_test_pred: .4f}")
        plt.scatter(x,y,color="blue", label="Actual data")
        plt.plot(x,w*x + b, color="red", label="Learned Line")
        plt.xlabel("x")
        plt.ylabel("Y")
        plt.title("Linear Regression from Scratch")
        plt.legend()
        plt.show()
        
