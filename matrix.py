import numpy as np
import matplotlib.pyplot as plt

cm = np.array([
    [1897,  3,  3,  2,  2],
    [   2, 59, 18,  0, 15],
    [   0, 10, 419,100, 16],
    [   0,  0,  9, 86,  0],
    [   0,  2, 14,  0,147],
])

classes = ["W", "N1", "N2", "N3", "REM"]

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm, cmap="Blues")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Epochos skaičius", rotation=270, labelpad=15)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Prognozuota stadija")
ax.set_ylabel("Tikroji stadija")
ax.set_title("Confusion matrix – Transformer modelis")

# Automatinis teksto kontrastas
threshold = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(j, i, cm[i, j],
                ha="center", va="center",
                color=color, fontsize=11)

# Tvarkingas tinklelis
ax.set_xticks(np.arange(-.5, len(classes), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig("ttransformer_confusion_matrix_blues.png", dpi=300)
plt.show()
