import matplotlib.pyplot as plt 
import pandas as pd




def show(indices, title, ):
    plt.figure(figsize=(12, 3.5))
    for i, idx in enumerate(indices[:4]):
        plt.subplot(1, 4, i + 1)
        plt.imshow(X_test[idx])
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()