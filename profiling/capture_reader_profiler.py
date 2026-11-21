import cProfile
import pstats
import sys
import time

from datetime import datetime
from pathlib import Path


# Accéder au lecteur de capture
project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / "packet_analyser" / "src"))

from capture_reader import read_capture


def main():
    data_dir = project_dir / "data" / "raw"

    captures = sorted(
        path
        for path in data_dir.iterdir()
        if path.suffix.lower() in {".pcap", ".pcapng"}
    )

    if not captures:
        raise ValueError("Aucune capture trouvée.")

    # Dossier des résultats
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = project_dir / "results" / "profiling" / date
    output.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    measurements = []

    for path in captures:
        print(f"Lecture : {path.name}")

        # Mesurer le temps sans profileur
        start = time.perf_counter()
        packets, counts = read_capture(path)
        duration = time.perf_counter() - start

        measurements.append(
            f"{path.name} : {duration:.3f} s, "
            f"{counts['total'] / duration:.0f} paquets/s"
        )

        del packets

        # Examiner les fonctions appelées pendant une autre lecture
        packets, counts = profiler.runcall(read_capture, path)
        del packets

    # Enregistrer les temps et les débits
    (output / "timings.txt").write_text( "\n".join(measurements), encoding="utf-8",)

    # Enregistrer le détail des fonctions
    with (output / "profile.txt").open("w", encoding="utf-8") as report:
        pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(30)

    print(f"Résultats : {output}")


if __name__ == "__main__":
    main()