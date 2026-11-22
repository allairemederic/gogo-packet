from collections import Counter
from pathlib import Path
import pandas as pd
import subprocess

from scapy.all import (IP, IPv6, TCP, UDP, PcapReader, IPv6ExtHdrFragment)

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

RTP_COLUMNS = [
    "capture_id",
    "packet_id",
    "stream_id",
    "arrival_time",
    "sequence",
    "rtp_timestamp",
    "payload_type",
    "clock_hz",
    "clock_source",
]

def extract_rtp(data, expected_type):

    if len(data) < 12:
        return None

    version = data[0] >> 6
    payload_type = data[1] & 0x7F

    if version != 2 or payload_type != expected_type:
        return None

    # Vérifier les identifiants CSRC
    csrc_count = data[0] & 0x0F
    header_length = 12 + 4 * csrc_count

    if len(data) < header_length:
        return None

    # Vérifier extension RTP
    if data[0] & 0x10:
        if len(data) < header_length + 4:
            return None

        extension_length = int.from_bytes(data[header_length + 2:header_length + 4], "big",)

        header_length += 4 + 4 * extension_length

        if len(data) < header_length:
            return None

    return {
        "sequence": int.from_bytes(data[2:4], "big"),
        "rtp_timestamp": int.from_bytes(data[4:8], "big"),
        "ssrc": int.from_bytes(data[8:12], "big"),
        "payload_type": payload_type,
    }



def read_capture(path, rtp_settings=None):
    rows = []
    rtp_rows = []
    summary = Counter()
    capture_id = Path(path).stem

    with PcapReader(str(path)) as reader:
        for number, packet in enumerate(reader, start=1):
            summary["total"] += 1

            # Trouver la couche IPv4 ou IPv6
            ip = packet.getlayer(IP) or packet.getlayer(IPv6)

            if ip is None:
                summary["non_ip"] += 1
                continue

            version = int(ip.version)

            # Ignorer les fragments
            fragmented = (version == 4 and (ip.frag or ip.flags.MF)) or (version == 6 and IPv6ExtHdrFragment in ip)

            if fragmented:
                summary["fragmented"] += 1
                continue

            # Calculer la taille totale du paquet IP
            if version == 4:
                length = int(ip.len)
            else:
                if int(ip.plen) == 0:
                    summary["ipv6_zero_payload_or_jumbogram"] += 1
                    continue

                length = 40 + int(ip.plen)

            # Ignorer les paquets incomplets
            ip_bytes = bytes(ip)

            if len(ip_bytes) < length:
                summary["truncated"] += 1
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

                transport_bytes = bytes(transport)
                ip_headers = len(bytes(ip)) - len(bytes(transport))
                available = length - ip_headers

                src_port = int(transport.sport)
                dst_port = int(transport.dport)

                if isinstance(transport, TCP):
                    protocol = "TCP"

                    # Retirer les en-têtes IP et TCP
                    tcp_header = 4 * int(transport.dataofs)

                    if tcp_header < 20 or tcp_header > available:
                        summary["invalid_length"] += 1
                        continue

                    payload_length = available - tcp_header

                else:
                    protocol = "UDP"

                    udp_length = int(transport.len)

                    # Retirer l'en-tête UDP
                    if not 8 <= udp_length <= available:
                        summary["invalid_length"] += 1
                        continue

                    payload_length = udp_length - 8

                if not 0 <= payload_length <= length:
                    summary["invalid_length"] += 1
                    continue

                # Examiner la config RTL configurée
                if protocol == "UDP" and rtp_settings is not None:
                    selected = (
                        ip.src == rtp_settings["server_ip"]
                        and src_port == rtp_settings["server_port"]
                        and ip.dst == rtp_settings["local_ip"]
                        and dst_port == rtp_settings["local_port"]
                    )

                    if selected:
                        data = transport_bytes[8:udp_length]

                        rtp = extract_rtp(data, rtp_settings["payload_type"])

                        if rtp is not None:
                            stream = (
                                f"{ip.src}:{src_port}->"
                                f"{ip.dst}:{dst_port}/"
                                f"ssrc={rtp['ssrc']}"
                            )

                            rtp_rows.append({
                                "capture_id": capture_id,
                                "packet_id": number,
                                "stream_id": stream,
                                "arrival_time": float(packet.time),
                                "sequence": rtp["sequence"],
                                "rtp_timestamp": rtp["rtp_timestamp"],
                                "payload_type": rtp["payload_type"],
                                "clock_hz": rtp_settings["clock_hz"],
                                "clock_source": rtp_settings["clock_source"],
                            })

                            summary["rtp_selected"] += 1


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

            summary["retained"] += 1

    packets = pd.DataFrame(rows, columns=COLUMNS)
    rtp_packets = pd.DataFrame(rtp_rows, columns=RTP_COLUMNS)

    return packets, dict(summary), rtp_packets


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