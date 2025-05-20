import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === USTAWIENIA ===
INPUT_FILE = "test_optimized_results.csv"
OUTPUT_DIR = "plots"

# === WCZYTANIE I PRZYGOTOWANIE DANYCH ===
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Heatmapa: MSE względem liczby warstw i neuronów (dla każdego LR osobno)
for lr_val in sorted(df["learning_rate"].unique()):
    df_lr = df[df["learning_rate"] == lr_val]
    df_reduced = df_lr.groupby(["layers", "neurons"], as_index=False)["loss"].min()
    pivot = df_reduced.pivot(index="layers", columns="neurons", values="loss")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", linewidths=0.5)
    plt.title(f"MSE (loss) | learning_rate = {lr_val}")
    plt.xlabel("Neuronów w warstwie")
    plt.ylabel("Liczba warstw")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_loss_lr_{str(lr_val).replace('.', '_')}.png")
    plt.close()


#Heatmapa: learning_rate vs layers (dla każdej liczby neuronów)
for n_val in sorted(df["neurons"].unique()):
    df_n = df[df["neurons"] == n_val]
    df_reduced = df_n.groupby(["learning_rate", "layers"], as_index=False)["loss"].min()
    pivot = df_reduced.pivot(index="learning_rate", columns="layers", values="loss")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"MSE (loss) | neurons = {n_val}")
    plt.xlabel("Liczba warstw")
    plt.ylabel("Learning rate")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_lr_vs_layers_neurons_{n_val}.png")
    plt.close()


# Wykresy: loss vs epochs — po jednym wykresie na każde learning_rate
unique_lrs = sorted(df["learning_rate"].unique())
unique_neurons = sorted(df["neurons"].unique())

for lr in unique_lrs:
    plt.figure(figsize=(10, 6))
    df_lr = df[df["learning_rate"] == lr]

    for neurons in unique_neurons:
        df_n = df_lr[df_lr["neurons"] == neurons]

        # znajdź konfigurację (layers) z najniższym minimalnym loss
        grouped = df_n.groupby("layers").agg({"loss": "min"}).reset_index()
        if grouped.empty:
            continue
        best_layers = grouped.sort_values("loss").iloc[0]["layers"]

        # wybierz dane tylko dla najlepszej konfiguracji
        subset = df_n[df_n["layers"] == best_layers]

        if len(subset["epochs"].unique()) < 2:
            continue

        label = f"{neurons} neu (L={int(best_layers)})"
        sns.lineplot(data=subset, x="epochs", y="loss", label=label)

    plt.title(f"Loss vs Epochs | learning_rate = {lr}")
    plt.xlabel("Epoki")
    plt.ylabel("Loss (MSE)")
    plt.legend(title="Neuronów (najlepsza liczba warstw)")
    plt.tight_layout()
    fname = f"loss_vs_epochs_simplified_lr_{str(lr).replace('.', '_')}.png"
    plt.savefig(f"{OUTPUT_DIR}/{fname}")
    plt.close()

#Wykres punktowy: czas vs jakość
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="time", y="loss", hue="learning_rate", palette="tab10")
plt.title("Czas vs Jakość (Loss)")
plt.xlabel("Czas treningu [s]")
plt.ylabel("Loss (MSE)")
plt.legend(title="Learning rate")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/scatter_time_vs_loss.png")
plt.close()
