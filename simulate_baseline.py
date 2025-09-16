"""
Train Traffic Optimizer - Baseline SimPy Simulator
Simulates train congestion on a single-track section with visualization
"""

import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Configuration
RANDOM_SEED = 42
NUM_STATIONS = 6
STATIONS = [f"S{i}" for i in range(1, NUM_STATIONS + 1)]
SINGLE_TRACK_START = "S3"
SINGLE_TRACK_END = "S4"
NUM_TRAINS_EB = 4  # Eastbound trains
NUM_TRAINS_WB = 4  # Westbound trains
HEADWAY_MIN = 3    # Minutes between train departures
RUN_TIME_MIN = 4   # Minutes to run between stations
DWELL_TIME_MIN = 1 # Minutes to dwell at stations
SINGLE_TRACK_CAPACITY = 1  # Only 1 train can be on single track at a time
CLEARANCE_TIME_MIN = 2  # Minutes between trains on single track

# Delay parameters
PRIMARY_DELAY_MEAN = 1.0
PRIMARY_DELAY_STD = 0.5
ENROUTE_DELAY_PROB = 0.3
ENROUTE_DELAY_MAX = 2.0

OUTPUT_DIR = "outputs"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class Train:
    train_id: str
    direction: str  # 'EB' or 'WB'
    origin: str
    destination: str
    planned_departure: float
    actual_departure: float = 0.0
    arrival_times: Dict[str, float] = None
    departure_times: Dict[str, float] = None
    delays: List[float] = None
    train_states: Dict[str, str] = None
    
    def __post_init__(self):
        if self.arrival_times is None:
            self.arrival_times = {}
        if self.departure_times is None:
            self.departure_times = {}
        if self.delays is None:
            self.delays = []
        if self.train_states is None:
            self.train_states = {}


class TrainTrafficSimulator:
    def __init__(self):
        self.env = simpy.Environment()
        self.single_track = simpy.Resource(self.env, capacity=SINGLE_TRACK_CAPACITY)
        self.station_resources = {station: simpy.Resource(self.env, capacity=1) for station in STATIONS}
        
        # Metrics tracking
        self.trains: List[Train] = []
        self.queue_lengths: List[Tuple[float, int]] = []  # (time, queue_length)
        self.throughput_times: List[float] = []
        self.delays: List[float] = []
        
        # Visualization data
        self.train_positions: Dict[str, float] = {}  # train_id -> position (0-1)
        self.train_states: Dict[str, str] = {}  # train_id -> state
        self.visualization_times: List[float] = []
        self.visualization_data: List[Dict] = []
        
    def generate_primary_delay(self) -> float:
        """Generate primary delay using normal distribution"""
        delay = np.random.normal(PRIMARY_DELAY_MEAN, PRIMARY_DELAY_STD)
        return max(0.0, delay)
    
    def generate_enroute_delay(self) -> float:
        """Generate en-route delay"""
        if random.random() < ENROUTE_DELAY_PROB:
            return random.uniform(0, ENROUTE_DELAY_MAX)
        return 0.0
    
    def create_trains(self) -> List[Train]:
        """Create train schedule"""
        trains = []
        start_time = 8 * 60  # 8:00 AM
        
        # Eastbound trains (S1 -> S6)
        for i in range(NUM_TRAINS_EB):
            train = Train(
                train_id=f"EB{i+1}",
                direction="EB",
                origin=STATIONS[0],
                destination=STATIONS[-1],
                planned_departure=start_time + i * HEADWAY_MIN
            )
            trains.append(train)
        
        # Westbound trains (S6 -> S1)
        for i in range(NUM_TRAINS_WB):
            train = Train(
                train_id=f"WB{i+1}",
                direction="WB",
                origin=STATIONS[-1],
                destination=STATIONS[0],
                planned_departure=start_time + i * HEADWAY_MIN
            )
            trains.append(train)
        
        return trains
    
    def get_train_path(self, train: Train) -> List[str]:
        """Get the path for a train"""
        if train.direction == "EB":
            return STATIONS
        else:
            return list(reversed(STATIONS))
    
    def get_position_on_line(self, train: Train, current_station: str) -> float:
        """Get train position as a value between 0 and 1"""
        path = self.get_train_path(train)
        try:
            station_index = path.index(current_station)
            return station_index / (len(path) - 1)
        except ValueError:
            return 0.0
    
    def train_process(self, train: Train):
        """Simulate a single train's journey"""
        # Primary delay at origin
        primary_delay = self.generate_primary_delay()
        yield self.env.timeout(primary_delay)
        train.actual_departure = self.env.now
        train.delays.append(primary_delay)
        
        path = self.get_train_path(train)
        train.train_states[train.train_id] = "departing"
        
        for i, station in enumerate(path):
            # Arrive at station
            train.arrival_times[station] = self.env.now
            train.train_states[train.train_id] = f"at_{station}"
            
            # Update position for visualization
            self.train_positions[train.train_id] = self.get_position_on_line(train, station)
            
            # Dwell at station (except at origin and destination)
            if i > 0 and i < len(path) - 1:
                with self.station_resources[station].request() as req:
                    yield req
                    dwell_delay = self.generate_enroute_delay()
                    yield self.env.timeout(DWELL_TIME_MIN + dwell_delay)
                    train.delays.append(dwell_delay)
            
            train.departure_times[station] = self.env.now
            
            # Move to next station
            if i < len(path) - 1:
                next_station = path[i + 1]
                
                # Check if this is the single track segment
                is_single_track = (station == SINGLE_TRACK_START and next_station == SINGLE_TRACK_END) or \
                                (station == SINGLE_TRACK_END and next_station == SINGLE_TRACK_START)
                
                if is_single_track:
                    # Record queue length before entering single track
                    queue_length = len(self.single_track.queue)
                    self.queue_lengths.append((self.env.now, queue_length))
                    
                    # Request single track
                    with self.single_track.request() as req:
                        yield req
                        
                        # Record queue length after entering
                        queue_length = len(self.single_track.queue)
                        self.queue_lengths.append((self.env.now, queue_length))
                        
                        # Run on single track
                        run_delay = self.generate_enroute_delay()
                        yield self.env.timeout(RUN_TIME_MIN + run_delay + CLEARANCE_TIME_MIN)
                        train.delays.append(run_delay)
                        
                        # Record throughput
                        self.throughput_times.append(self.env.now)
                else:
                    # Regular segment
                    run_delay = self.generate_enroute_delay()
                    yield self.env.timeout(RUN_TIME_MIN + run_delay)
                    train.delays.append(run_delay)
                
                train.train_states[train.train_id] = f"running_{station}_to_{next_station}"
        
        # Train completed journey
        train.train_states[train.train_id] = "completed"
        self.train_positions[train.train_id] = 1.0 if train.direction == "EB" else 0.0
        
        # Calculate total delay
        total_delay = sum(train.delays)
        self.delays.append(total_delay)
    
    def run_simulation(self):
        """Run the complete simulation"""
        self.trains = self.create_trains()
        
        # Start all trains
        for train in self.trains:
            self.env.process(self.train_process(train))
        
        # Run simulation
        self.env.run(until=12 * 60)  # Run until 12:00 PM
    
    def calculate_metrics(self) -> Dict:
        """Calculate simulation metrics"""
        if not self.trains:
            return {}
        
        # Average delay
        avg_delay = np.mean(self.delays) if self.delays else 0.0
        
        # Trains per hour
        if self.throughput_times:
            time_span = max(self.throughput_times) - min(self.throughput_times)
            trains_per_hour = len(self.throughput_times) * 60 / time_span if time_span > 0 else 0
        else:
            trains_per_hour = 0.0
        
        # Max queue length
        max_queue_length = max([ql for _, ql in self.queue_lengths]) if self.queue_lengths else 0
        
        # Average queue length
        if self.queue_lengths:
            avg_queue_length = np.mean([ql for _, ql in self.queue_lengths])
        else:
            avg_queue_length = 0.0
        
        return {
            "average_delay_minutes": round(avg_delay, 2),
            "trains_per_hour": round(trains_per_hour, 2),
            "max_queue_length": max_queue_length,
            "average_queue_length": round(avg_queue_length, 2),
            "total_trains": len(self.trains),
            "total_delays": len(self.delays)
        }
    
    def create_visualization(self):
        """Create animated visualization of train congestion"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: Train positions over time
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, len(self.trains) + 1)
        ax1.set_xlabel('Position on Line (S1 to S6)')
        ax1.set_ylabel('Train ID')
        ax1.set_title('Train Positions Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add station markers
        for i, station in enumerate(STATIONS):
            pos = i / (len(STATIONS) - 1)
            ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
            ax1.text(pos, len(self.trains) + 0.5, station, ha='center', va='bottom')
        
        # Highlight single track section
        s3_pos = STATIONS.index(SINGLE_TRACK_START) / (len(STATIONS) - 1)
        s4_pos = STATIONS.index(SINGLE_TRACK_END) / (len(STATIONS) - 1)
        ax1.axvspan(s3_pos, s4_pos, alpha=0.2, color='red', label='Single Track')
        
        # Bottom plot: Queue length over time
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Queue Length')
        ax2.set_title('Queue Length at Single Track Over Time')
        ax2.grid(True, alpha=0.3)
        
        if self.queue_lengths:
            times, queue_lengths = zip(*self.queue_lengths)
            ax2.step(times, queue_lengths, where='post', linewidth=2)
            ax2.fill_between(times, 0, queue_lengths, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/baseline_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, metrics: Dict):
        """Save simulation results to files"""
        import os
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Save metrics
        with open(f'{OUTPUT_DIR}/baseline_metrics.txt', 'w') as f:
            f.write("Baseline Simulation Results\n")
            f.write("=" * 30 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        # Save detailed train data
        with open(f'{OUTPUT_DIR}/baseline_train_details.csv', 'w') as f:
            f.write("train_id,direction,planned_departure,actual_departure,total_delay\n")
            for train in self.trains:
                total_delay = sum(train.delays)
                f.write(f"{train.train_id},{train.direction},{train.planned_departure},{train.actual_departure},{total_delay:.2f}\n")
        
        # Save queue timeline
        with open(f'{OUTPUT_DIR}/baseline_queue_timeline.csv', 'w') as f:
            f.write("time_minutes,queue_length\n")
            for time, queue_len in self.queue_lengths:
                f.write(f"{time:.2f},{queue_len}\n")


def main():
    """Main simulation function"""
    print("Starting Baseline Train Traffic Simulation...")
    print("=" * 50)
    
    # Create and run simulator
    simulator = TrainTrafficSimulator()
    simulator.run_simulation()
    
    # Calculate and display metrics
    metrics = simulator.calculate_metrics()
    
    print("\nSimulation Results:")
    print("-" * 20)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Save results
    simulator.save_results(metrics)
    
    # Create visualization
    print("\nCreating visualization...")
    simulator.create_visualization()
    
    print(f"\nResults saved to '{OUTPUT_DIR}/' directory")
    print("Visualization saved as 'baseline_visualization.png'")


if __name__ == "__main__":
    main()
