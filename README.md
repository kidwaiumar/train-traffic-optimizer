# Train Traffic Optimizer - Toy Congestion Scenario

This project simulates a small 6-station line with a single-track bottleneck and 8 tightly scheduled trains. It injects random delays and shows how queues form, reducing throughput and increasing delays.

## Quick start

1) Run the simulation (no external dependencies required)
```
python simulate.py
```

Outputs are written to `outputs/`:
- `timetable_planned.csv`: planned schedule by train and station
- `timetable_actual.csv`: simulated actual times
- `events_log.csv`: resource entry/exit and dwell events
- `metrics.csv`: key metrics (avg delay, max queue length, throughput)
- `queue_timeline.csv`: queue length over time at the single-track
- `queue_length.txt`: ASCII plot of queue over time
- `throughput_times.txt`: times trains cleared the bottleneck
- `terminal_delay_histogram.txt`: list of terminal delays (min)

## Scenario details
- 6 stations: S1 — S2 — S3 — S4 — S5 — S6
- Single-track bottleneck: between S3 and S4
- 8 trains total: 4 eastbound (S1→S6) and 4 westbound (S6→S1)
- Tight headways to create conflict pressure
- Random primary delays injected on departure and en-route

You can tweak parameters at the top of `simulate.py` to explore different demand, delay, and capacity settings.
