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


def plot_histogram(values, title, xlabel, path, fits=None, bins=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not len(values):
        return

    if bins is None:
        bins = get_bins([values])

    figure, axis = plt.subplots()

    _, edges, _ = axis.hist(values, bins=bins, density=True, edgecolor="black", alpha=0.5)

    for fit in fits or []:
        if "distribution" not in fit:
            continue

        distribution = getattr(stats, fit["distribution"])
        parameters = fit["parameters"]

        if fit.get("data_type") == "discrete":

            # Probabilité théorique dans chaque classe de l'histogramme
            lower = np.ceil(edges[:-1]) - 1
            upper = np.ceil(edges[1:]) - 1
            upper[-1] = np.floor(edges[-1])

            masses = (distribution.cdf(upper, **parameters) - distribution.cdf(lower, **parameters))

            axis.stairs(masses / np.diff(edges), edges, label=fit["distribution"], linewidth=2)

        else:
            grid = np.linspace(values.min(), values.max(), 500)
            axis.plot(grid, distribution.pdf(grid, **parameters), label=fit["distribution"])

    axis.set(title=f"{title} — n={len(values)}", xlabel=xlabel, ylabel="Densité")

    if fits and any("distribution" in fit for fit in fits):
        axis.legend()

    axis.ticklabel_format(axis="y", style="plain", useOffset=False)

    figure.tight_layout()
    figure.savefig(path)

    plt.close(figure)


def plot_cdf(values, title, xlabel, path, fits):

    # Proportion des valeurs inférieures ou égales à x
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not len(values):
        return

    points, counts = np.unique(values, return_counts=True)

    figure, axis = plt.subplots()

    axis.step(points, np.cumsum(counts) / len(values), where="post", label="Observée")

    for fit in fits:
        if "distribution" not in fit:
            continue

        distribution = getattr(stats, fit["distribution"])

        grid = np.linspace( values.min(), values.max(), 2000)

        if fit.get("data_type") == "discrete":
            grid = np.unique(np.rint(grid))

        axis.plot(
            grid,
            distribution.cdf(grid,**fit["parameters"]),
            label=(f"{fit['distribution']} " f"(p={fit['p_value']:.3g})"),
        )

    axis.set(
        title=title,
        xlabel=xlabel,
        ylabel="Probabilité cumulée",
        ylim=(0, 1.02),
    )

    axis.legend()

    figure.tight_layout()
    figure.savefig(path)

    plt.close(figure)


def plot_wifi_ethernet(
    populations,
    activity,
    xlabel,
    path,
    bins=None,
):
    # Format :
    # {
    #     "wifi": {"youtube_wifi_01": valeurs, ...},
    #     "ethernet": {"youtube_ethernet_01": valeurs, ...},
    # }
    groups = {}

    for connection in ["wifi", "ethernet"]:
        groups[connection] = {}

        for capture, values in populations.get(connection, {}).items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]

            if len(values):
                groups[connection][capture] = values

    # Attendre des données pour les deux connexions
    if not groups["wifi"] or not groups["ethernet"]:
        return False

    if bins is None:
        bins = get_bins(
            values
            for captures in groups.values()
            for values in captures.values()
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
    )

    colors = {
        "wifi": "tab:blue",
        "ethernet": "tab:orange",
    }

    names = {
        "wifi": "Wi-Fi",
        "ethernet": "Ethernet",
    }

    styles = ["-", "--", ":"]

    # Garder chaque répétition visible
    for connection, captures in groups.items():
        for index, (capture, values) in enumerate(
            sorted(captures.items())
        ):
            label = (
                f"{names[connection]} — {capture} "
                f"(n={len(values)})"
            )

            style = styles[index % len(styles)]

            axes[0].hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=colors[connection],
                linestyle=style,
                linewidth=1.5,
            )

            points, counts = np.unique(
                values,
                return_counts=True,
            )

            axes[1].step(
                points,
                np.cumsum(counts) / len(values),
                where="post",
                color=colors[connection],
                linestyle=style,
                label=label,
            )

    axes[0].set(
        title="Histogrammes — classes communes",
        ylabel="Densité",
    )

    axes[1].set(
        title="Distributions cumulées",
        ylabel="Probabilité cumulée",
        ylim=(0, 1.02),
    )

    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.2)

    axes[1].legend(fontsize=8)

    figure.suptitle(
        f"{activity} — comparaison Wi-Fi / Ethernet"
    )

    figure.tight_layout()
    figure.savefig(path)

    plt.close(figure)

    return True


def plot_means(summary, path):
    if summary.empty:
        return

    variables = summary["variable"].unique()

    figure, axes = plt.subplots(
        len(variables),
        1,
        figsize=(9, 3 * len(variables)),
        squeeze=False,
    )

    for axis, variable in zip(axes[:, 0], variables):
        selected = summary.loc[
            summary["variable"].eq(variable)
        ]

        axis.bar(
            selected["capture_id"],
            selected["mean"],
        )

        axis.set(
            title=variable,
            ylabel="Moyenne",
        )

        axis.tick_params(
            axis="x",
            labelrotation=45,
        )

    figure.tight_layout()
    figure.savefig(path)

    plt.close(figure)


def plot_classification(matrix, classes, scores, output):
    # Matrice de confusion sur le test.
    display = ConfusionMatrixDisplay(
        matrix,
        display_labels=classes,
    )

    display.plot(
        xticks_rotation=45,
        colorbar=False,
    )

    display.figure_.tight_layout()

    display.figure_.savefig(
        output / "confusion_matrix.pdf"
    )

    plt.close(display.figure_)

    # Comparer les modèles et la référence sur validation.
    figure, axis = plt.subplots()

    axis.bar(
        list(scores),
        list(scores.values()),
    )

    axis.tick_params(
        axis="x",
        labelrotation=35,
    )

    axis.set(
        ylabel="F1 macro de validation",
        ylim=(0, 1),
    )

    figure.tight_layout()

    figure.savefig(
        output / "model_comparison.pdf"
    )

    plt.close(figure)


def plot_tcp_sizes(
    data,
    path,
    title="Tailles IP par classe TCP",
):
    # Utiliser l'entraînement pour explorer les caractéristiques.
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

    # Histogrammes avec les mêmes bornes.
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

    # Médiane, quartiles et points au-delà des moustaches.
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
    # Décrire le test après évaluation.
    plot_tcp_sizes(
        test,
        output / "tcp_sizes_by_class.pdf",
        "Test — tailles IP par classe TCP",
    )

    # Les scores de régression ne sont pas des probabilités.
    figure, axis = plt.subplots()

    correct = test["correct"].to_numpy()

    axis.scatter(
        test["ip_length"],
        test["prediction_score"],
        c=np.where(correct, "tab:blue", "tab:red"),
        s=8,
        alpha=0.4,
    )

    axis.axhline(
        0.5,
        color="black",
        linestyle="--",
        label="Seuil de décision",
    )

    axis.set(
        xlabel="Taille IP (octets)",
        ylabel="Score de régression",
        title="Test : bleu = correct, rouge = erreur",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output / "test_prediction_scores.pdf"
    )

    plt.close(figure)


def plot_jitter(jitter, path):
    if jitter.empty:
        return

    data = jitter.copy()

    # Retrouver le type de connexion dans le nom de capture
    data["connection"] = data["capture_id"].str.extract(r"_(wifi|ethernet)_", expand=False)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        sharey=True,
    )

    for axis, connection in zip(axes, ["wifi", "ethernet"]):
        selected = data.loc[data["connection"].eq(connection)]

        for (capture, stream), packets in selected.groupby(["capture_id", "stream_id"]):
            packets = packets.sort_values("arrival_time", kind="stable")

            # Commencer chaque courbe à zéro seconde
            elapsed = (packets["arrival_time"] - packets["arrival_time"].iloc[0])

            axis.plot(
                elapsed,
                packets["jitter_seconds"] * 1000,
                linewidth=0.8,
                label=f"{capture} (n={len(packets)})",
            )

        axis.set(
            title="Wi-Fi" if connection == "wifi" else "Ethernet",
            ylabel="Gigue RTP (ms)",
        )

        axis.grid(alpha=0.3)

        if not selected.empty:
            axis.legend(fontsize=8)

    axes[-1].set_xlabel("Temps écoulé depuis la première estimation du flux (s)")

    figure.suptitle("Gigue RTP entrante — horloge supposée à 48 kHz")

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)