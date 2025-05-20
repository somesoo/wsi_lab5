import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj dane
df = pd.read_csv("test_optimized_results.csv")  # <- Podmień na nazwę swojego pliku CSV

# Wybieramy tylko ostatnie epoki
df_last = df.sort_values("epochs").groupby(["layers", "neurons", "learning_rate"]).last().reset_index()

# Lista unikalnych wartości learning rate
lr_values = sorted(df_last["learning_rate"].unique())

# Tworzymy figure i 6 osi (bo chcemy 3 w pierwszym rzędzie, 3 w drugim – ostatni subplot pusty)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, lr in enumerate(lr_values):
    ax = axes[i]
    subset = df_last[df_last["learning_rate"] == lr]
    sns.barplot(
        data=subset,
        x="layers",
        y="time",
        hue="neurons",
        ax=ax,
        palette="viridis"
    )
    ax.set_title(f"Learning rate = {lr}")
    ax.set_xlabel("Liczba warstw")
    ax.set_ylabel("Czas [s]")

# Usuwamy ostatni pusty subplot (nr 5 → 6. miejsce z indeksu 5)
fig.delaxes(axes[-1])

# Ustawiamy wspólną legendę
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Neurony", loc="upper center", ncol=len(labels))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/czas_centrowany_barplot.png")
plt.show()