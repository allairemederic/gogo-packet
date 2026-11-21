import json
import shutil

from datetime import datetime
from pathlib import Path

import pandas as pd

from capture_reader import read_capture, detect_quic
from packet_cleaner import clean_packets, prepare_classification_data

from statistics import (
    compute_moments,
    compute_interarrival_times,
    fit_distributions,
    compute_jitter,
)

from plot_creator import (
    plot_histogram,
    plot_comparison,
    plot_distribution_cdf,
)

from classification import train_and_evaluate


def save_json(path, content):
    path.write_text(
        json.dumps(
            content,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def analyze_capture(capture, packets, figures_dir):
    # Sélectionner les populations à étudier.
    tcp_data = packets.loc[
        packets["protocol"].eq("TCP")
        & packets["payload_length"].gt(0)
    ]

    udp = packets.loc[packets["protocol"].eq("UDP")]

    populations = {
        "ip_length_bytes": (
            packets["ip_length"].to_numpy(),
            "Taille IP (octets)",
        ),
        "tcp_data_interarrival_seconds": (
            compute_interarrival_times(tcp_data),
            "Intervalle TCP avec payload (s)",
        ),
        "udp_observed_interarrival_seconds": (
            compute_interarrival_times(udp),
            "Intervalle UDP observé, contrôle compris (s)",
        ),
    }

    rows = []
    fits = {}

    for variable, (values, xlabel) in populations.items():
        rows.append({
            "capture_id": capture,
            "variable": variable,
            **compute_moments(values),
        })

        fits[variable] = fit_distributions(
            values,
            discrete=(variable == "ip_length_bytes"),
        )

        plot_histogram(
            values,
            capture,
            xlabel,
            figures_dir / f"{capture}_{variable}.pdf",
            fits[variable],
        )

        plot_distribution_cdf(
            values,
            capture,
            xlabel,
            figures_dir / f"{capture}_{variable}_cdf.pdf",
            fits[variable],
        )

    return rows, fits


def main():
    # Dossiers du projet.
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data" / "raw"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_dir}"
        )

    captures = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".pcap", ".pcapng"}
    )

    if not captures:
        raise ValueError("Aucune capture trouvée.")

    if len({path.stem for path in captures}) != len(captures):
        raise ValueError(
            "Les captures doivent avoir des noms distincts."
        )

    # Créer le dossier de cette exécution.
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    output = (
        project_dir
        / "results"
        / "packet_analysis"
        / date
    )

    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)

    frames = []
    capture_summary = {}

    tshark_available = shutil.which("tshark") is not None

    # Lire les captures.
    for path in captures:
        print(f"Lecture : {path.name}")

        frame, counts = read_capture(path)

        info = {
            "path": str(path),
            "counts": counts,
        }

        # Reconnaître QUIC si TShark est installé.
        if tshark_available:
            identifiers = detect_quic(path)

            frame["is_quic"] = frame["packet_id"].isin(
                identifiers
            )

            info["quic"] = {
                "status": "checked",
                "recognized_retained_packets": int(
                    frame["is_quic"].sum()
                ),
            }

        else:
            frame["is_quic"] = pd.NA

            info["quic"] = {
                "status": "not_checked",
                "reason": "TShark non installé.",
            }

        frames.append(frame)
        capture_summary[path.stem] = info

    # Regrouper et nettoyer les paquets.
    packets = clean_packets(
        pd.concat(frames, ignore_index=True)
    )

    if packets.empty:
        save_json(
            output / "summary.json",
            {
                "captures": capture_summary,
                "status": "no_usable_packets",
            },
        )

        raise ValueError(
            "Aucun paquet exploitable. Voir summary.json."
        )

    packets.to_csv(
        output / "packets.csv",
        index=False,
    )

    # Analyser chaque capture séparément.
    rows = []
    fits = {}

    for capture, selected in packets.groupby("capture_id"):
        print(f"Statistiques : {capture}")

        capture_rows, capture_fits = analyze_capture(
            capture,
            selected,
            figures_dir,
        )

        rows.extend(capture_rows)
        fits[capture] = capture_fits

        capture_summary[capture]["protocol_counts"] = {
            name: int(count)
            for name, count
            in selected["protocol"].value_counts().items()
        }

        capture_summary[capture]["retained_after_cleaning"] = (
            len(selected)
        )

    # Calculer la gigue si un CSV RTP validé est disponible.
    rtp_path = project_dir / "data" / "rtp.csv"

    jitter_info = {
        "status": "not_computed",
        "reason": "Aucun CSV RTP validé dans data/rtp.csv.",
    }

    if rtp_path.is_file():
        jitter = compute_jitter(rtp_path)

        jitter_info = {
            "status": (
                "computed"
                if not jitter.empty
                else "insufficient_data"
            ),
            "source": str(rtp_path),
        }

        groups = jitter.groupby(["capture_id", "stream_id"])

        for index, ((capture, stream), selected) in enumerate(groups):
            values = selected["jitter_seconds"].to_numpy()

            variable = f"rtp_jitter_seconds_stream_{stream}"

            rows.append({
                "capture_id": capture,
                "variable": variable,
                **compute_moments(values),
            })

            jitter_fits = fit_distributions(values)

            fits.setdefault(capture, {})[variable] = jitter_fits

            plot_histogram(
                values,
                f"{capture} — {stream}",
                "Gigue RTP (s)",
                figures_dir / f"jitter_{index}.pdf",
                jitter_fits,
            )

    # Comparer les moyennes.
    table = pd.DataFrame(rows)

    plot_comparison(
        table,
        figures_dir / "comparison.pdf",
    )

    # Convertir les valeurs manquantes en null pour le JSON.
    records = (
        table.astype(object)
        .where(table.notna(), None)
        .to_dict(orient="records")
    )

    save_json(
        output / "summary.json",
        {
            "captures": capture_summary,
            "cleaned_packet_count": len(packets),
            "analysis_scope": {
                "sizes": (
                    "Tous les paquets IP conservés, "
                    "offloading inchangé."
                ),
                "tcp_arrivals": (
                    "TCP avec payload, deux directions "
                    "confondues, par capture."
                ),
                "udp_arrivals": (
                    "Tous les UDP observés, "
                    "données et contrôle non séparés."
                ),
                "udp_classification": (
                    "Non effectuée : aucune étiquette "
                    "fiable données/contrôle."
                ),
                "application_filter": (
                    "Aucun : le trafic de fond peut être présent."
                ),
                "quic": (
                    "Reconnaissance par TShark seulement; "
                    "aucune étiquette données/contrôle QUIC."
                ),
            },
            "statistics": records,
            "distribution_fits": fits,
            "jitter": jitter_info,
        },
    )

    # Classification TCP.
    print("Classification TCP avec/sans payload...")

    classification_data = prepare_classification_data(packets)

    report = train_and_evaluate(
        classification_data,
        output,
    )

    if report["status"] != "completed":
        print(report["reason"])

    print(
        f"{len(packets)} paquets après nettoyage. "
        f"Résultats : {output}"
    )


if __name__ == "__main__":
    main()