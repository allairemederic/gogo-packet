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
    get_bins,
    plot_histogram,
    plot_cdf,
    plot_wifi_ethernet,
    plot_means,
    plot_jitter,
)

from classification import train_and_evaluate

from rtp_settings import RTP_SETTINGS


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


def get_values(packets):

    # Une seule capture à la fois
    tcp_data = packets.loc[packets["protocol"].eq("TCP") & packets["payload_length"].gt(0)]
    udp = packets.loc[packets["protocol"].eq("UDP")]

    return {
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


def analyze_capture(
    capture,
    packets,
    populations,
    bins,
    figures_dir,
):
    metadata = {
        "activity": packets["activity"].iloc[0],
        "connection": packets["connection"].iloc[0],
        "repetition": int(packets["repetition"].iloc[0]),
    }

    rows = []
    fits = {}

    for variable, (values, xlabel) in populations.items():
        rows.append({
            "capture_id": capture,
            **metadata,
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
            bins=bins[variable],
        )

        plot_cdf(
            values,
            capture,
            xlabel,
            figures_dir / f"{capture}_{variable}_cdf.pdf",
            fits[variable],
        )

    return rows, fits


def main():

    project_dir = Path(__file__).resolve().parents[2]
    pcap_dir = project_dir / "data" / "pcap"
    rtp_dir = project_dir / "data" / "rtp"

    if not pcap_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {pcap_dir}")

    captures = sorted(
        path
        for path in pcap_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".pcap", ".pcapng"}
    )

    if not captures:
        raise ValueError("Aucune capture trouvée.")

    if len({path.stem for path in captures}) != len(captures):
        raise ValueError("Les captures doivent avoir des noms distincts.")

    # Créer le dossier des résultats
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
    rtp_frames = []
    capture_summary = {}

    tshark_available = shutil.which("tshark") is not None

    # Lire les captures et extraire le RTP
    for path in captures:
        print(f"Lecture : {path.name}")

        # Paramètres RTP
        settings = RTP_SETTINGS.get(path.stem)

        frame, counts, rtp_frame = read_capture(path, rtp_settings=settings)

        info = {
            "path": str(path),
            "counts": counts,
            "rtp": {
                "status": (
                    "checked"
                    if settings is not None
                    else "not_configured"
                ),
                "selected_packets": len(rtp_frame),
            },
        }

        if settings is not None:
            info["rtp"]["settings"] = settings

        if not rtp_frame.empty:
            rtp_frames.append(rtp_frame)

        # Reconnaître QUIC si TShark est installé
        if tshark_available:
            identifiers = detect_quic(path)

            frame["is_quic"] = frame["packet_id"].isin(identifiers)

            info["quic"] = {
                "status": "checked",
                "recognized_retained_packets": int(frame["is_quic"].sum()),
            }

        else:
            frame["is_quic"] = pd.NA
            info["quic"] = {"status": "not_checked", "reason": "TShark non installé.",}

        frames.append(frame)
        capture_summary[path.stem] = info

    # Nettoyer les paquets et ajouter les métadonnées
    packets = clean_packets(pd.concat(frames, ignore_index=True))

    if packets.empty:
        save_json(
            output / "summary.json",
            {
                "captures": capture_summary,
                "status": "no_usable_packets",
            },
        )

        raise ValueError("Aucun paquet exploitable. Voir summary.json.")

    packets.to_csv(output / "packets.csv", index=False)

    rows = []
    fits = {}
    histogram_bins = {}
    comparisons = {}

    # Comparer les captures d'une même activité
    for activity, activity_packets in packets.groupby("activity"):
        groups = {
            capture: selected
            for capture, selected
            in activity_packets.groupby("capture_id")
        }

        # Calculer les valeurs séparément pour chaque capture
        populations = {
            capture: get_values(selected)
            for capture, selected in groups.items()
        }

        first = next(iter(populations.values()))

        # Utiliser des classes communes pour les histogrammes
        bins = {
            variable: get_bins(
                values[variable][0]
                for values in populations.values()
            )
            for variable in first
        }

        histogram_bins[activity] = {
            variable: (
                edges.tolist()
                if edges is not None
                else None
            )
            for variable, edges in bins.items()
        }

        # Statistiques et figures individuelles
        for capture, selected in groups.items():
            print(f"Statistiques : {capture}")

            capture_rows, capture_fits = analyze_capture(
                capture,
                selected,
                populations[capture],
                bins,
                figures_dir,
            )

            rows.extend(capture_rows)
            fits[capture] = capture_fits

            capture_summary[capture].update({
                "activity": activity,
                "connection": selected["connection"].iloc[0],
                "repetition": int(
                    selected["repetition"].iloc[0]
                ),
                "protocol_counts": {
                    name: int(count)
                    for name, count
                    in selected["protocol"].value_counts().items()
                },
                "retained_after_cleaning": len(selected),
            })

        # Comparaisons Wi-Fi/Ethernet
        comparisons[activity] = {}

        for variable, (_, xlabel) in first.items():
            connections = {
                "wifi": {},
                "ethernet": {},
            }

            for capture, selected in groups.items():
                connection = selected["connection"].iloc[0]

                connections[connection][capture] = (
                    populations[capture][variable][0]
                )

            created = plot_wifi_ethernet(
                connections,
                activity,
                xlabel,
                figures_dir / (
                    f"{activity}_wifi_ethernet_{variable}.pdf"
                ),
                bins=bins[variable],
            )

            comparisons[activity][variable] = (
                "created"
                if created
                else "missing_or_empty_connection"
            )

    # Calculer la gigue avec le RTP extrait pendant cette exécution
    jitter_info = {
        "status": "not_computed",
        "reason": (
            "Aucun paquet RTP extrait des communications configurées."
        ),
    }

    if rtp_frames:
        rtp_packets = pd.concat(
            rtp_frames,
            ignore_index=True,
        )

        # Enregistrer les données RTP dans data/rtp/.
        rtp_dir.mkdir(parents=True, exist_ok=True)
        rtp_path = rtp_dir / "rtp.csv"

        rtp_packets.to_csv(
            rtp_path,
            index=False,
        )

        print(
            f"Gigue : {len(rtp_packets)} paquets RTP extraits."
        )

        jitter = compute_jitter(rtp_path)

        plot_jitter(jitter, figures_dir / "zoom_jitter_time.pdf",)

        jitter_info = {
            "status": (
                "computed_exploratory"
                if not jitter.empty
                else "insufficient_data"
            ),
            "source": str(rtp_path),
            "rtp_packets": len(rtp_packets),
            "captures": sorted(
                rtp_packets["capture_id"].unique().tolist()
            ),
            "direction": (
                "Entrante sur l'ordinateur de capture."
            ),
            "clock_note": (
                "Horloge supposée à 48 000 Hz, compatible avec "
                "les horodatages observés ; codec non confirmé."
            ),
        }

        streams = jitter.groupby(
            ["capture_id", "stream_id"]
        )

        for index, ((capture, stream), selected) in enumerate(streams):
            values = selected["jitter_seconds"].to_numpy()
            variable = f"rtp_jitter_seconds_stream_{stream}"

            # Ajouter les métadonnées de la capture
            metadata = capture_summary[capture]

            rows.append({
                "capture_id": capture,
                "activity": metadata["activity"],
                "connection": metadata["connection"],
                "repetition": metadata["repetition"],
                "variable": variable,
                **compute_moments(values),
            })

            jitter_fits = fit_distributions(values)
            fits.setdefault(capture, {})[variable] = jitter_fits

            plot_histogram(
                values,
                f"{capture} — gigue RTP entrante",
                "Gigue RTP (s)",
                figures_dir / f"jitter_{index}.pdf",
                jitter_fits,
            )

            plot_cdf(
                values,
                f"{capture} — gigue RTP entrante",
                "Gigue RTP (s)",
                figures_dir / f"jitter_{index}_cdf.pdf",
                jitter_fits,
            )

    # Comparer les moyennes
    table = pd.DataFrame(rows)

    plot_means(
        table,
        figures_dir / "comparison.pdf",
    )

    # Convertir les valeurs manquantes en null pour le JSON
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
                    "Reconnaissance par TShark seulement ; "
                    "aucune étiquette données/contrôle QUIC."
                ),
                "rtp": (
                    "Extraction limitée aux communications "
                    "configurées ; les autres flux ne sont "
                    "pas automatiquement identifiés."
                ),
            },
            "statistics": records,
            "distribution_fits": fits,
            "histogram_bins": histogram_bins,
            "wifi_ethernet_comparisons": comparisons,
            "jitter": jitter_info,
        },
    )

    # Classification TCP
    print("Classification TCP avec/sans payload...")

    classification_data = prepare_classification_data(packets)

    report = train_and_evaluate(classification_data,output,)

    if report["status"] != "completed":
        print(report["reason"])

    print(
        f"{len(packets)} paquets après nettoyage. "
        f"Résultats : {output}"
    )


if __name__ == "__main__":
    main()