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
from plot_creator import plot_histograms
from classification import train_and_evaluate
from rtp_settings import RTP_SETTINGS


def save_json(path, content):
    # Enregistrer un fichier JSON
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

    tcp_data = packets.loc[packets["protocol"].eq("TCP") & packets["payload_length"].gt(0)]
    udp = packets.loc[packets["protocol"].eq("UDP")]

    return {
        "ip_length_bytes": packets["ip_length"].to_numpy(),
        "tcp_data_interarrival_seconds": (compute_interarrival_times(tcp_data)),
        "udp_observed_interarrival_seconds": (compute_interarrival_times(udp)),
    }

# Calculer les statistiques séparément pour chaque capture
def analyze_capture(capture, packets):
    
    rows = []
    fits = {}

    metadata = {
        "capture_id": capture,
        "activity": packets["activity"].iloc[0],
        "connection": packets["connection"].iloc[0],
        "repetition": int(packets["repetition"].iloc[0]),
    }

    for variable, values in get_values(packets).items():
        rows.append({
            **metadata,
            "variable": variable,
            **compute_moments(values),
        })

        fits[variable] = fit_distributions(values, discrete=(variable == "ip_length_bytes"))

    return rows, fits


def plot_report_figures(packets, rtp_packets, jitter, figures_dir):
    
    connections = {"Wi-Fi": "wifi", "Ethernet": "ethernet"}
    activities = {"YouTube": "youtube", "LAPACK": "lapack", "Zoom": "zoom"}

    # Figure 1 : tailles IP, avec un panneau par activité
    sizes = {}

    for title, activity in activities.items():
        sizes[title] = {}

        for name, connection in connections.items():
            selected = packets.loc[packets["activity"].eq(activity) & packets["connection"].eq(connection)]

            sizes[title][name] = (selected["ip_length"].to_numpy())

    plot_histograms(sizes, "Taille IP (octets)", figures_dir / "ip_sizes.pdf")

    # Figure 2 : intervalles TCP et RTP
    arrivals = {}

    for title, activity in activities.items():
        panel = f"{title} — TCP avec charge utile"
        arrivals[panel] = {}

        for name, connection in connections.items():
            selected = packets.loc[
                packets["activity"].eq(activity)
                & packets["connection"].eq(connection)
                & packets["protocol"].eq("TCP")
                & packets["payload_length"].gt(0)
            ]

            intervals = []

            # Ne pas créer d'intervalle entre deux captures
            for _, capture in selected.groupby("capture_id"):
                intervals.extend(compute_interarrival_times(capture) * 1000)

            arrivals[panel][name] = intervals

    if not rtp_packets.empty:
        arrivals["Zoom — RTP entrant"] = {}

        for name, connection in connections.items():
            selected = rtp_packets.loc[rtp_packets["capture_id"].str.contains(f"_{connection}_", regex=False)]

            intervals = []

            # Ne pas mélanger les intervalles de plusieurs flux
            for _, stream in selected.groupby(["capture_id", "stream_id"]):
                stream = stream.rename(columns={"arrival_time": "timestamp"})

                intervals.extend(compute_interarrival_times(stream) * 1000)

            arrivals["Zoom — RTP entrant"][name] = intervals

    plot_histograms(arrivals,"Intervalle d'arrivée (ms)", figures_dir / "packet_arrivals.pdf")

    # Figure 3 : gigue RTP, avec un panneau par répétition
    if not jitter.empty:
        jitter_panels = {}

        for repetition in [1, 2, 3]:
            title = f"Répétition {repetition:02d}"
            jitter_panels[title] = {}

            for name, connection in connections.items():

                capture = (f"zoom_{connection}_{repetition:02d}")
                selected = jitter.loc[jitter["capture_id"].eq(capture)]
                jitter_panels[title][name] = (selected["jitter_seconds"].to_numpy() * 1000)

        plot_histograms(jitter_panels, "Gigue RTP (ms)", figures_dir / "rtp_jitter.pdf")


def main():

    # Dossiers du projet
    project_dir = Path(__file__).resolve().parents[2]
    pcap_dir = project_dir / "data" / "pcap"
    rtp_dir = project_dir / "data" / "rtp"

    if not pcap_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {pcap_dir}")

    # Trouver les captures
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

    # Créer le dossier de cette exécution
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

    # Lire les captures et extraire les informations RTP
    for path in captures:
        print(f"Lecture : {path.name}")

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

            info["quic"] = {
                "status": "not_checked",
                "reason": "TShark non installé.",
            }

        frames.append(frame)
        capture_summary[path.stem] = info

    # Nettoyer les paquets et ajouter les métadonnées
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

        raise ValueError("Aucun paquet exploitable. Voir summary.json.")

    packets.to_csv(output / "packets.csv", index=False)

    rows = []
    fits = {}

    # Statistiques IP, TCP et UDP par captur
    for capture, selected in packets.groupby("capture_id"):
        print(f"Statistiques : {capture}")

        capture_rows, capture_fits = analyze_capture(capture, selected,)

        rows.extend(capture_rows)
        fits[capture] = capture_fits

        capture_summary[capture].update({
            "activity": selected["activity"].iloc[0],
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

    # Initialiser les données RTP et de gigue
    rtp_packets = pd.DataFrame()
    jitter = pd.DataFrame()

    jitter_info = {
        "status": "not_computed",
        "reason": (
            "Aucun paquet RTP extrait des communications configurées."
        ),
    }

    if rtp_frames:
        rtp_packets = pd.concat(rtp_frames, ignore_index=True)

        # Enregistrer uniquement le RTP de cette exécution
        rtp_dir.mkdir(parents=True, exist_ok=True)
        rtp_path = rtp_dir / "rtp.csv"

        rtp_packets.to_csv(rtp_path, index=False)

        # Arrivées RTP, séparées par capture et par flux
        for (capture, stream), selected in rtp_packets.groupby(["capture_id", "stream_id"]):
            arrivals = selected.rename(
                columns={"arrival_time": "timestamp"}
            )

            intervals = compute_interarrival_times(arrivals)

            metadata = capture_summary[capture]
            variable = "rtp_data_interarrival_seconds"

            rows.append({
                "capture_id": capture,
                "stream_id": stream,
                "activity": metadata["activity"],
                "connection": metadata["connection"],
                "repetition": metadata["repetition"],
                "variable": variable,
                **compute_moments(intervals),
            })

            fit_name = f"{variable}_stream_{stream}"

            fits.setdefault(capture, {})[fit_name] = (fit_distributions(intervals))

        # Calculer la gigue RTP
        print( f"Gigue : {len(rtp_packets)} paquets RTP extraits.")

        jitter = compute_jitter(rtp_path)

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
                "les horodatages observés. Codec non confirmé."
            ),
        }

        # Statistiques de gigue par capture et par flux
        for (capture, stream), selected in jitter.groupby(
            ["capture_id", "stream_id"]
        ):
            values = selected["jitter_seconds"].to_numpy()

            metadata = capture_summary[capture]
            variable = f"rtp_jitter_seconds_stream_{stream}"

            rows.append({
                "capture_id": capture,
                "stream_id": stream,
                "activity": metadata["activity"],
                "connection": metadata["connection"],
                "repetition": metadata["repetition"],
                "variable": variable,
                **compute_moments(values),
            })

            fits.setdefault(capture, {})[variable] = (
                fit_distributions(values)
            )

    # Créer les figures statistiques regroupées
    print("Création des figures statistiques...")

    plot_report_figures(packets, rtp_packets, jitter, figures_dir)

    # Remplacer les valeurs manquantes par null dans le JSON
    table = pd.DataFrame(rows)

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
                    "TCP avec charge utile, toutes connexions "
                    "et deux directions regroupées par capture."
                ),
                "udp_arrivals": (
                    "Tous les UDP observés, "
                    "données et contrôle non séparés."
                ),
                "udp_classification": (
                    "Non effectuée. Aucune étiquette "
                    "fiable données/contrôle."
                ),
                "application_filter": (
                    "Aucun. Le trafic de fond peut être présent."
                ),
                "quic": (
                    "Reconnaissance par TShark seulement. "
                    "Aucune étiquette données/contrôle QUIC."
                ),
                "rtp": (
                    "Extraction limitée aux communications configurées."
                ),
                "rtp_arrivals": (
                    "RTP entrant sélectionné, RTCP exclu. "
                    "Intervalles calculés par capture et par flux."
                ),
                "figures": (
                    "Les répétitions sont regroupées pour les "
                    "histogrammes IP et d'arrivées seulement. "
                    "Les statistiques restent séparées par capture. "
                    "Chaque panneau utilise des classes communes "
                    "pour Wi-Fi et Ethernet."
                ),
            },
            "statistics": records,
            "distribution_fits": fits,
            "jitter": jitter_info,
        },
    )

    # Classification TCP et figures ML
    print("Classification TCP avec/sans charge utile...")

    classification_data = prepare_classification_data(packets)

    report = train_and_evaluate(classification_data, output)

    if report["status"] != "completed":
        print(report["reason"])

    print(
        f"{len(packets)} paquets après nettoyage. "
        f"Résultats : {output}"
    )


if __name__ == "__main__":
    main()