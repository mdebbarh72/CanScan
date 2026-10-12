import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter



class Painter():

    @classmethod
    def drawBarPlt(data, title, xl, yl):
        counts = Counter(data)

        plt.figure(figsize=(10, 5))
        plt.bar(counts.keys(), counts.values())
        plt.title(title)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

    @classmethod
    def displayImageSimples(images, classes, labels):
        fig, axes = plt.subplots(
            1, len(classes),
            figsize=(4 * len(classes), 4)
        )

        axes = np.atleast_1d(axes)

        for ax, class_name in zip(axes, classes):
            indices = np.flatnonzero(labels == class_name)

            if len(indices) == 0:
                ax.set_visible(False)
                continue

            idx = indices[0]

            ax.imshow(images[idx])
            ax.set_title(class_name)
            ax.axis("off")

        plt.tight_layout()
        plt.show()