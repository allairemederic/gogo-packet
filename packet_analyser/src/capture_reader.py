from collections import Counter
from pathlib import Path
import pandas as pd
import subprocess

from scapy.all import IP, IPv6, TCP, UDP, PcapReader, IPv6ExtHdrFragment

COLUMNS = [
    "capture_id",
    "packet_id",
    "timestamp",
    "ip_version",
    "ip_length",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "payload_length",
]


def read_capture(path):
    rows = []
    audit = Counter()
    capture_id = Path(path).stem

    with PcapReader(str(path)) as reader:
        for number, packet in enumerate(reader, start=1):
            audit["total"] += 1

            # Trouver la couche IPv4 ou IPv6
            ip = packet.getlayer(IP) or packet.getlayer(IPv6)

            if ip is None:
                audit["non_ip"] += 1
                continue

            version = int(ip.version)

            # Ignorer les fragments
            fragmented = (version == 4 and (ip.frag or ip.flags.MF)) or (
                version == 6 and IPv6ExtHdrFragment in ip
            )

            if fragmented:
                audit["fragmented"] += 1
                continue

            # Calculer la taille totale du paquet IP
            if version == 4:
                length = int(ip.len)
            else:
                if int(ip.plen) == 0:
                    audit["ipv6_zero_payload_or_jumbogram"] += 1
                    continue

                length = 40 + int(ip.plen)

            # Ignorer les paquets incomplets
            ip_bytes = bytes(ip)
            if len(ip_bytes) < length:
                audit["truncated"] += 1
                continue

            transport = ip.payload

            # Passer les en-têtes supplémentaires IPv6
            while version == 6 and transport.name.startswith("IPv6 Extension"):
                transport = transport.payload

            protocol = "OTHER"
            src_port = 0
            dst_port = 0
            payload_length = None

            if isinstance(transport, (TCP, UDP)):
                available = length - (len(ip_bytes) - len(bytes(transport)))
                src_port = int(transport.sport)
                dst_port = int(transport.dport)

                if isinstance(transport, TCP):
                    protocol = "TCP"

                    # Retirer les en-têtes IP et TCP
                    ip_headers = len(bytes(ip)) - len(bytes(transport))
                    tcp_header = 4 * int(transport.dataofs)
                    if tcp_header < 20 or tcp_header > available:
                        audit["invalid_length"] += 1
                        continue

                    payload_length = length - ip_headers - tcp_header

                else:
                    protocol = "UDP"

                    # Retirer l'en-tête UDP
                    if not 8 <= int(transport.len) <= available:
                        audit["invalid_length"] += 1
                        continue
                    payload_length = int(transport.len) - 8

                if not 0 <= payload_length <= length:
                    audit["invalid_length"] += 1
                    continue

            rows.append(
                [
                    capture_id,
                    number,
                    float(packet.time),
                    version,
                    length,
                    ip.src,
                    ip.dst,
                    src_port,
                    dst_port,
                    protocol,
                    payload_length,
                ]
            )

            audit["retained"] += 1

    return pd.DataFrame(rows, columns=COLUMNS), dict(audit)


def detect_quic(path):
    """Retourner les numéros des paquets reconnus comme QUIC."""

    command = [
        "tshark",
        "-n",  # Ne pas résoudre les noms réseau.
        "-r",
        str(path),  # Lire la capture.
        "-Y",
        "quic",  # Sélectionner les paquets QUIC.
        "-T",
        "fields",
        "-e",
        "frame.number",  # Extraire leur numéro.
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return {int(number) for number in result.stdout.splitlines() if number.strip()}