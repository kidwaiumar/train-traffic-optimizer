import os
import math
import random
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------------------------
# Parameters (tweakable)
# ---------------------------
RANDOM_SEED = 42
NUM_STATIONS = 6  # S1..S6
STATIONS = [f"S{i}" for i in range(1, NUM_STATIONS + 1)]
SINGLE_TRACK_SEGMENT = ("S3", "S4")  # bottleneck between S3 and S4
NUM_TRAINS_EB = 4  # eastbound S1->S6
NUM_TRAINS_WB = 4  # westbound S6->S1
HEADWAY_MIN = 3  # planned headway in minutes at origin departures
DWELL_MIN = 1  # planned dwell at each intermediate station
RUN_MIN_PER_SEGMENT = 4  # planned run time between adjacent stations (per segment)
CLEARANCE_MIN = 2  # minimum separation on single track (entry headway)
PRIMARY_DELAY_MEAN = 1.0  # minutes
PRIMARY_DELAY_STD = 1.0   # minutes
ENROUTE_DELAY_PROB = 0.3  # chance of an en-route small delay on each segment
ENROUTE_DELAY_MAX = 2.0   # max extra minutes per segment when a delay occurs

OUTPUT_DIR = "outputs"


@dataclass
class TrainPlan:
    train_id: str
    direction: str  # 'EB' or 'WB'
    origin: str
    destination: str
    planned_departure_min: float


@dataclass
class Event:
    time_min: float
    train_id: str
    event_type: str  # 'enter_segment', 'exit_segment', 'dwell_start', 'dwell_end'
    location: str    # station or segment label


random.seed(RANDOM_SEED)


def normal_pos(mean: float, std: float) -> float:
    # Positive normal via Box-Muller; clamp at zero
    # Use two uniforms in (0,1]
    u1 = 1.0 - random.random()
    u2 = 1.0 - random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    val = mean + std * z0
    return max(0.0, val)


def build_planned_schedule() -> Tuple[List[TrainPlan], List[Dict[str, float]]]:
    plans: List[TrainPlan] = []

    # Schedule tightly with headways starting 08:00
    t0 = 8 * 60
    for i in range(NUM_TRAINS_EB):
        plans.append(TrainPlan(
            train_id=f"EB{i+1}",
            direction="EB",
            origin=STATIONS[0],
            destination=STATIONS[-1],
            planned_departure_min=t0 + i * HEADWAY_MIN,
        ))
    for i in range(NUM_TRAINS_WB):
        plans.append(TrainPlan(
            train_id=f"WB{i+1}",
            direction="WB",
            origin=STATIONS[-1],
            destination=STATIONS[0],
            planned_departure_min=t0 + i * HEADWAY_MIN,
        ))

    rows: List[Dict[str, float]] = []
    for plan in plans:
        path = STATIONS if plan.direction == "EB" else list(reversed(STATIONS))
        current_time = plan.planned_departure_min
        rows.append({
            "train_id": plan.train_id,
            "direction": plan.direction,
            "station": path[0],
            "planned_arrival_min": float('nan'),
            "planned_departure_min": current_time,
        })
        for si in range(len(path) - 1):
            seg_to = path[si + 1]
            current_time += RUN_MIN_PER_SEGMENT
            rows.append({
                "train_id": plan.train_id,
                "direction": plan.direction,
                "station": seg_to,
                "planned_arrival_min": current_time,
                "planned_departure_min": current_time + (0 if seg_to == path[-1] else DWELL_MIN),
            })
            if seg_to != path[-1]:
                current_time += DWELL_MIN

    return plans, rows


def enroute_delay() -> float:
    if random.random() < ENROUTE_DELAY_PROB:
        return random.uniform(0.0, ENROUTE_DELAY_MAX)
    return 0.0


def simulate(plans: List[TrainPlan], planned_rows: List[Dict[str, float]]):
    s_left, s_right = SINGLE_TRACK_SEGMENT
    single_track_available_min = 0.0

    events: List[Event] = []

    # Build per-train path
    station_paths: Dict[str, List[str]] = {p.train_id: (STATIONS if p.direction == "EB" else list(reversed(STATIONS))) for p in plans}

    # Primary departure delays
    dep_delay: Dict[str, float] = {p.train_id: normal_pos(PRIMARY_DELAY_MEAN, PRIMARY_DELAY_STD) for p in plans}

    # Seed origin departures
    train_station_times: Dict[str, Dict[str, Dict[str, float]]] = {
        p.train_id: {s: {"arr": float('nan'), "dep": float('nan')} for s in station_paths[p.train_id]} for p in plans
    }

    # Planned departures lookup
    planned_dep_lookup: Dict[Tuple[str, str], float] = {}
    for r in planned_rows:
        if r["station"] == (STATIONS[-1] if r["direction"] == "EB" else STATIONS[0]):
            # terminal row, skip
            pass
        # origin row: planned_departure_min exists and arrival is NaN
        key = (r["train_id"], r["station"])  # includes origin too
        planned_dep_lookup[key] = r.get("planned_departure_min", float('nan'))

    for p in plans:
        origin = station_paths[p.train_id][0]
        planned_dep = planned_dep_lookup.get((p.train_id, origin), p.planned_departure_min)
        actual_dep = planned_dep + dep_delay[p.train_id]
        train_station_times[p.train_id][origin]["dep"] = actual_dep
        train_station_times[p.train_id][origin]["arr"] = actual_dep
        events.append(Event(actual_dep, p.train_id, "dwell_end", origin))

    # Build segment requests
    segment_requests: List[Tuple[float, str, Tuple[str, str]]] = []
    for p in plans:
        path = station_paths[p.train_id]
        current_time = train_station_times[p.train_id][path[0]]["dep"]
        for i in range(len(path) - 1):
            segment_requests.append((current_time, p.train_id, (path[i], path[i + 1])))
            current_time += RUN_MIN_PER_SEGMENT + (0 if path[i + 1] == path[-1] else DWELL_MIN)

    segment_requests.sort(key=lambda x: x[0])

    queue_timeline: List[Tuple[float, int]] = []
    queue_count = 0
    throughput_times: List[float] = []

    def record_queue(t: float):
        queue_timeline.append((t, queue_count))

    for requested_time, train_id, seg in segment_requests:
        from_s, to_s = seg
        is_single_track = set(seg) == set(SINGLE_TRACK_SEGMENT)
        path = station_paths[train_id]
        idx_from = path.index(from_s)

        # Ensure arrival/dep at from_s
        if math.isnan(train_station_times[train_id][from_s]["arr"]):
            if idx_from == 0:
                train_station_times[train_id][from_s]["arr"] = train_station_times[train_id][from_s]["dep"]
            else:
                prev_s = path[idx_from - 1]
                prev_dep = train_station_times[train_id][prev_s]["dep"]
                if math.isnan(prev_dep):
                    prev_dep = train_station_times[train_id][prev_s]["arr"]
                run_time = RUN_MIN_PER_SEGMENT + enroute_delay()
                arr_time = prev_dep + run_time
                train_station_times[train_id][from_s]["arr"] = arr_time
                if from_s != path[-1]:
                    train_station_times[train_id][from_s]["dep"] = arr_time + DWELL_MIN
                    events.append(Event(arr_time, train_id, "dwell_start", from_s))
                    events.append(Event(arr_time + DWELL_MIN, train_id, "dwell_end", from_s))
        if math.isnan(train_station_times[train_id][from_s]["dep"]) and from_s != path[-1]:
            train_station_times[train_id][from_s]["dep"] = train_station_times[train_id][from_s]["arr"] + DWELL_MIN

        earliest_entry = max(train_station_times[train_id][from_s]["dep"], requested_time)

        if is_single_track:
            if earliest_entry < single_track_available_min:
                queue_count += 1
                record_queue(earliest_entry)
                entry_time = single_track_available_min
                queue_count -= 1
                record_queue(entry_time)
            else:
                entry_time = earliest_entry
                record_queue(entry_time)
            run_time = RUN_MIN_PER_SEGMENT + enroute_delay()
            exit_time = entry_time + run_time
            single_track_available_min = exit_time + CLEARANCE_MIN
            throughput_times.append(exit_time)
        else:
            run_time = RUN_MIN_PER_SEGMENT + enroute_delay()
            entry_time = earliest_entry
            exit_time = entry_time + run_time

        events.append(Event(entry_time, train_id, "enter_segment", f"{from_s}->{to_s}"))
        events.append(Event(exit_time, train_id, "exit_segment", f"{from_s}->{to_s}"))

        train_station_times[train_id][to_s]["arr"] = exit_time
        if to_s != path[-1]:
            train_station_times[train_id][to_s]["dep"] = exit_time + DWELL_MIN
            events.append(Event(exit_time, train_id, "dwell_start", to_s))
            events.append(Event(exit_time + DWELL_MIN, train_id, "dwell_end", to_s))

    # Build outputs
    timetable_actual_rows: List[Dict[str, float]] = []
    for p in plans:
        path = station_paths[p.train_id]
        for s in path:
            timetable_actual_rows.append({
                "train_id": p.train_id,
                "direction": p.direction,
                "station": s,
                "actual_arrival_min": train_station_times[p.train_id][s]["arr"],
                "actual_departure_min": train_station_times[p.train_id][s]["dep"],
            })

    # Planned rows already built
    events_rows = [e.__dict__ for e in events]

    # Metrics
    dest_by_train = {p.train_id: (STATIONS[-1] if p.direction == "EB" else STATIONS[0]) for p in plans}

    # index planned and actual by (train, station)
    planned_index: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in planned_rows:
        planned_index[(r["train_id"], r["station"])] = r
    actual_index: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in timetable_actual_rows:
        actual_index[(r["train_id"], r["station"])] = r

    terminal_delays = []
    for p in plans:
        terminal = dest_by_train[p.train_id]
        planned_arr = planned_index[(p.train_id, terminal)]["planned_arrival_min"]
        actual_arr = actual_index[(p.train_id, terminal)]["actual_arrival_min"]
        terminal_delays.append(actual_arr - planned_arr)

    def mean(vals: List[float]) -> float:
        vals2 = [v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))]
        return sum(vals2) / len(vals2) if vals2 else float('nan')

    avg_arrival_delay = mean(terminal_delays)
    max_arrival_delay = max(terminal_delays) if terminal_delays else 0.0

    max_queue_len = max([ql for _, ql in queue_timeline], default=0)

    throughput_times_sorted = sorted(throughput_times)
    total_throughput = len(throughput_times_sorted)
    if len(throughput_times_sorted) >= 2:
        horizon_min = throughput_times_sorted[-1] - throughput_times_sorted[0]
        throughput_per_hour = 60.0 * total_throughput / horizon_min if horizon_min > 0 else float('nan')
    else:
        throughput_per_hour = 0.0

    metrics = {
        "avg_terminal_arrival_delay_min": round(avg_arrival_delay, 2) if not math.isnan(avg_arrival_delay) else "nan",
        "max_terminal_arrival_delay_min": round(max_arrival_delay, 2),
        "max_queue_length": int(max_queue_len),
        "total_throughput": int(total_throughput),
        "throughput_per_hour": round(throughput_per_hour, 2) if not isinstance(throughput_per_hour, float) or not math.isnan(throughput_per_hour) else "nan",
    }

    return planned_rows, timetable_actual_rows, events_rows, queue_timeline, throughput_times_sorted, metrics, terminal_delays


def ensure_output_dir():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, float]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_outputs(planned_rows, actual_rows, events_rows, queue_timeline, throughput_times, metrics, terminal_delays):
    ensure_output_dir()

    write_csv(
        os.path.join(OUTPUT_DIR, "timetable_planned.csv"),
        ["train_id", "direction", "station", "planned_arrival_min", "planned_departure_min"],
        planned_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "timetable_actual.csv"),
        ["train_id", "direction", "station", "actual_arrival_min", "actual_departure_min"],
        actual_rows,
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "events_log.csv"),
        ["time_min", "train_id", "event_type", "location"],
        events_rows,
    )

    # metrics.csv
    with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # queue timeline csv
    with open(os.path.join(OUTPUT_DIR, "queue_timeline.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_min", "queue_length"])
        for t, ql in queue_timeline:
            writer.writerow([t, ql])

    # simple ascii plots
    with open(os.path.join(OUTPUT_DIR, "queue_length.txt"), "w", encoding="utf-8") as f:
        f.write("Queue length over time (ASCII)\n")
        for t, ql in queue_timeline:
            f.write(f"{t:7.2f} | " + ("#" * ql) + f" ({ql})\n")

    with open(os.path.join(OUTPUT_DIR, "throughput_times.txt"), "w", encoding="utf-8") as f:
        f.write("Throughput times across single-track (min)\n")
        for t in throughput_times:
            f.write(f"{t:.2f}\n")

    with open(os.path.join(OUTPUT_DIR, "terminal_delay_histogram.txt"), "w", encoding="utf-8") as f:
        f.write("Terminal arrival delays (min):\n")
        for d in terminal_delays:
            f.write(f"{d:.2f}\n")


def main():
    plans, planned_rows = build_planned_schedule()
    planned_rows, actual_rows, events_rows, queue_timeline, throughput_times, metrics, terminal_delays = simulate(plans, planned_rows)
    save_outputs(planned_rows, actual_rows, events_rows, queue_timeline, throughput_times, metrics, terminal_delays)

    print("Avg terminal arrival delay (min):", metrics["avg_terminal_arrival_delay_min"])
    print("Max terminal arrival delay (min):", metrics["max_terminal_arrival_delay_min"])
    print("Max queue length:", metrics["max_queue_length"]) 
    print("Total throughput:", metrics["total_throughput"]) 
    print("Throughput (per hour):", metrics["throughput_per_hour"])
    print(f"Outputs written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
