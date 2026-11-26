import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay


def get_bins(groups, count=50):

    # Construire une grille commune pour les captures comparées
    arrays = []

    for values in groups:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values):
            arrays.append(values)

    if not arrays:
        return None

    return np.histogram_bin_edges(np.concatenate(arrays), bins=count)



def plot_histograms(panels, xlabel, path):

    columns = 3 if len(panels) == 3 else 2
    rows = (len(panels) + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False)

    colors = {"Wi-Fi": "tab:blue", "Ethernet": "tab:orange",}

    for axis, (title, groups) in zip(axes.flat, panels.items()):
        # Retirer les valeurs non valides
        groups = {
            name: np.asarray(values, dtype=float)
            for name, values in groups.items()
        }

        groups = {
            name: values[np.isfinite(values)]
            for name, values in groups.items()
        }

        available = [
            values for values in groups.values()
            if len(values)
        ]

        axis.set_title(title,fontsize=18, pad=12)

        if not available:
            axis.text(
                0.5, 0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                fontsize=16,
                transform=axis.transAxes,
            )
            continue

        # Utiliser les mêmes classes pour les deux connexions
        bins = np.histogram_bin_edges(
            np.concatenate(available),
            bins=40,
        )

        for name, values in groups.items():
            if not len(values):
                continue

            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=colors[name],
                label=f"{name}\n(n = {len(values):,})".replace(",", " "),
            )

        axis.set_xlabel(xlabel, fontsize=17)
        axis.set_ylabel("Densité", fontsize=17)

        axis.tick_params(axis="both", labelsize=14)

        axis.locator_params(axis="x", nbins=4)
        axis.locator_params(axis="y", nbins=8)


        axis.set_axisbelow(True)
        axis.grid(axis="y", linestyle="--", alpha=0.4)

        axis.set_ylim(bottom=0)
        axis.margins(y=0.12)

        axis.legend(fontsize=12, loc="upper right")

        axis.xaxis.get_offset_text().set_fontsize(13)
        axis.yaxis.get_offset_text().set_fontsize(13)

    # Masquer les panneaux inutilisés
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)

    figure.tight_layout(pad=1.5, w_pad=2)
    figure.savefig(path)
    plt.close(figure)


def plot_classification(matrix, classes, scores, output):
    # Présenter les classes sur deux lignes
    class_names = [
        name.replace("payload", "charge utile")
            .replace("charge utile", "charge\nutile")
        for name in classes
    ]

    # Matrice de confusion
    figure, axis = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        xticks_rotation=0,
        colorbar=False,
        values_format="d",
        text_kw={"fontsize": 18},
    )

    axis.set_xlabel(
        "Classe prédite",
        fontsize=17,
        labelpad=12,
    )

    axis.set_ylabel(
        "Classe réelle",
        fontsize=17,
        labelpad=12,
    )

    axis.tick_params(
        axis="both",
        labelsize=15,
        labelrotation=0,
    )

    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.pdf")
    plt.close(figure)

    # Comparaison des modèles sur la validation
    figure, axis = plt.subplots(figsize=(12, 6))

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:gray",
    ]

    bars = axis.bar(
        list(scores),
        list(scores.values()),
        color=colors[:len(scores)],
        width=0.65,
    )

    axis.set_ylabel(
        "F1 macro de validation",
        fontsize=18,
    )

    axis.set_ylim(0, 1)
    axis.set_yticks(np.arange(0, 1.01, 0.1))

    axis.tick_params(
        axis="x",
        labelsize=14,
        labelrotation=0,
        pad=10,
    )

    axis.tick_params(
        axis="y",
        labelsize=15,
    )

    axis.set_axisbelow(True)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.bar_label(
        bars,
        labels=[
            f"{score:.3f}".replace(".", ",")
            for score in scores.values()
        ],
        padding=6,
        fontsize=15,
    )

    figure.tight_layout()
    figure.savefig(output / "model_comparison.pdf")
    plt.close(figure)


def plot_tcp_sizes(
    data,
    path,
    title="Tailles IP par classe TCP",
):
    # Utiliser l'entraînement pour explorer les caractéristiques
    groups = []
    labels = []

    for label, name in [
        (0, "Sans payload"),
        (1, "Avec payload"),
    ]:
        values = data.loc[
            data["label"].eq(label),
            "ip_length",
        ].to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        if len(values):
            groups.append(values)

            labels.append(
                f"{name} (n={len(values)})"
            )

    if not groups:
        return False

    bins = get_bins(groups)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    # Histogrammes avec les mêmes bornes
    for values, name in zip(groups, labels):
        axes[0].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=name,
        )

    axes[0].set(
        title="Histogrammes par classe",
        xlabel="Taille IP (octets)",
        ylabel="Densité",
    )

    axes[0].legend(fontsize=8)

    # Médiane, quartiles et points au-delà des moustaches
    positions = np.arange(1, len(groups) + 1)

    axes[1].boxplot(
        groups,
        positions=positions,
        widths=0.5,
        showfliers=True,
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)

    axes[1].set(
        title="Boxplots par classe",
        ylabel="Taille IP (octets)",
    )

    figure.suptitle(title)

    figure.tight_layout()
    figure.savefig(path)

    plt.close(figure)

    return True


def plot_test_results(test, output):

    # Comparer les tailles des deux classes sur le test
    plot_tcp_sizes(
        test,
        output / "tcp_sizes_by_class.pdf",
        "Test — tailles IP par classe TCP",
    )

    # Présenter les scores avec deux niveaux de détail
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    for axis, title in zip(
        axes,
        ["Vue complète", "Petits paquets"],
    ):
        # Tracer les erreurs en dernier pour les rendre visibles
        for correct, color, marker, name in [
            (True, "tab:blue", "o", "Correct"),
            (False, "tab:red", "x", "Erreur"),
        ]:
            selected = test.loc[
                test["correct"].eq(correct)
            ]

            axis.scatter(
                selected["ip_length"],
                selected["prediction_score"],
                color=color,
                marker=marker,
                s=35,
                alpha=0.7,
                label=name,
            )

        axis.axhline(
            0.5,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Seuil de décision",
        )

        axis.set_title(title, fontsize=16)
        axis.set_xlabel("Taille IP (octets)", fontsize=15)
        axis.set_ylabel("Score de régression", fontsize=15)

        axis.tick_params(
            axis="both",
            labelsize=12,
        )

        axis.set_axisbelow(True)
        axis.grid(
            linestyle="--",
            alpha=0.3,
        )

    # Agrandir la zone des petits paquets
    axes[1].set_xlim(40, 150)

    small_packets = test.loc[
        test["ip_length"].between(40, 150)
    ]

    if not small_packets.empty:
        # Garder visibles les scores et le seuil de décision
        low = min(
            float(small_packets["prediction_score"].min()),
            0.5,
        )

        high = max(
            float(small_packets["prediction_score"].max()),
            0.5,
        )

        margin = max(
            (high - low) * 0.2,
            0.001,
        )

        axes[1].set_ylim(
            low - margin,
            high + margin,
        )

        axes[1].ticklabel_format(
            axis="y",
            style="plain",
            useOffset=False,
        )

    axes[0].legend(fontsize=11)

    figure.tight_layout()

    figure.savefig(
        output / "test_prediction_scores.pdf"
    )

    plt.close(figure)