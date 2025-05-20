import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from matplotlib.lines import Line2D

# === USTAWIENIA ===
INPUT_FILE = "test_optimized_results.csv"
OUTPUT_FILE = "plots/loss_3d_final.png"

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
os.makedirs("plots", exist_ok=True)

# Grupowanie - najlepszy wynik dla każdego zestawu
df_reduced = df.groupby(["layers", "neurons", "learning_rate"], as_index=False)["loss"].min()

# Punkt z najmniejszym błędem
min_point = df_reduced[df_reduced["loss"] == df_reduced["loss"].min()].iloc[0]

# Kolory dyskretne dla learning_rate
unique_lrs = sorted(df_reduced["learning_rate"].unique())
lr_to_color = {lr: plt.cm.tab10(i % 10) for i, lr in enumerate(unique_lrs)}
colors = df_reduced["learning_rate"].map(lr_to_color)

# Tworzenie wykresu
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Oryginalne osie: X=layers, Y=neurons, Z=loss
ax.scatter(df_reduced["layers"], df_reduced["neurons"], df_reduced["loss"],
           c=colors, s=50, depthshade=True)

# Czerwony punkt: najlepszy wynik
ax.scatter(min_point["layers"], min_point["neurons"], min_point["loss"],
           color="red", s=120, marker="*", label="Najlepszy wynik")

# Opisy osi
ax.set_xlabel("Liczba warstw")
ax.set_ylabel("Neuronów w warstwie")
ax.set_zlabel("Loss (MSE)")
ax.set_title("Loss w zależności od architektury i learning rate")

# Legenda
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'LR={lr}',
           markerfacecolor=lr_to_color[lr], markersize=8) for lr in unique_lrs
]
legend_elements.append(Line2D([0], [0], marker='*', color='w', label='Najlepszy wynik',
                              markerfacecolor='red', markersize=12))
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.close()
