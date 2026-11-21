## Packet Analyser

cd packet_analyser/

sudo apt install tshark

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python src/main.py

## Profiling

python -m cProfile -o results/profiling/python.prof packet_analyser/src/main.py

python -m pstats results/profiling/python.prof
sort cumulative
stats 30
quit

python profiling/capture_reader_profiler.py