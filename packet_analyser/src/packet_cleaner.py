import numpy as np
import pandas as pd


def clean_packets(packets):
    data = packets.copy()

    # Convertir les valeurs en nombres
    columns = ["timestamp", "ip_length", "payload_length"]
    data[columns] = data[columns].apply(pd.to_numeric, errors="coerce")
    data[columns] = data[columns].replace([np.inf, -np.inf], np.nan)

    # Retirer les paquets sans taille ou sans horodatage valide
    data = data.dropna(subset=["timestamp", "ip_length"])
    data = data.loc[data["ip_length"] > 0].copy()

    # Vérifer le payload pour TCP et UDP
    transport = data["protocol"].isin(["TCP", "UDP"])
    valid_payload = data["payload_length"].between(0, data["ip_length"])
    data = data.loc[~transport | valid_payload].copy()

    # Extraire les informations du nom de capture
    # Ex: youtube_wifi_01
    pattern = (
        r"^(?P<activity>youtube|lapack|zoom)_"
        r"(?P<connection>wifi|ethernet)_"
        r"(?P<repetition>0[123])$"
    )

    metadata = (data["capture_id"].astype("string").str.extract(pattern))

    # Erreur si les noms ne respectent pas le format attendu
    invalid = metadata.isna().any(axis=1)

    if invalid.any():
        names = (data.loc[invalid, "capture_id"].astype(str).unique())

        raise ValueError("Nom de capture invalide : " + ", ".join(names))

    data["activity"] = metadata["activity"]
    data["connection"] = metadata["connection"]
    data["repetition"] = metadata["repetition"].astype(int)

    # Conserver les retransmissions et les numéros du PCAP
    return data.sort_values(["capture_id", "timestamp", "packet_id"], kind="stable").reset_index(drop=True)


def prepare_classification_data(packets):

    # Garder seulement TCP pour la classification
    data = packets.loc[packets["protocol"].eq("TCP")].copy()

    # 0 = sans payload
    # 1 = avec payload
    data["label"] = data["payload_length"].gt(0).astype(int)

    return data