import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd
import can
import paho.mqtt.client as mqtt

from simulator.can_publisher.codec import encode_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("csv_replayer")

MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
CAN_INTERFACE = os.environ.get("CAN_INTERFACE", "vcan0")
CAN_ENGINE_ID = os.environ.get("CAN_ENGINE_ID", "ENG01")

class CSVReplayer:
    def __init__(self, data_path, holdout_runs=None, hz=10):
        self.hz = hz
        self.data_path = data_path
        
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        if holdout_runs:
            with open(holdout_runs, "r") as f:
                conf = json.load(f)
                available = conf.get("available_runs", [])
                if available:
                    self.df = self.df[self.df["run_id"].isin(available)]
                    logger.info(f"Filtered to {len(available)} available runs.")
        
        # Pre-group by run_id
        self.runs = {run_id: group.sort_values("time_s") for run_id, group in self.df.groupby("run_id")}
        
        # Separate healthy and faulty runs
        # assuming 'none' or missing or 'healthy' means healthy
        # checking the data, fault_type seems to be 'none' when healthy.
        self.healthy_runs = []
        self.faulty_runs = {}
        
        for run_id, group in self.runs.items():
            faults = group["fault_type"].unique()
            real_faults = [f for f in faults if pd.notna(f) and f.lower() not in ("none", "healthy")]
            if not real_faults:
                self.healthy_runs.append(run_id)
            else:
                for f in real_faults:
                    if f not in self.faulty_runs:
                        self.faulty_runs[f] = []
                    self.faulty_runs[f].append(run_id)

        self.bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
        self.current_run_id = None
        self.current_run_idx = 0
        self.requested_fault = None
        
        self.mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        self.mqtt_client.loop_start()
        topic = f"uav/engine/{CAN_ENGINE_ID}/simulate/inject_fault"
        self.mqtt_client.subscribe(topic)
        logger.info(f"Subscribed to {topic}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            fault_type = payload.get("fault_type")
            if fault_type:
                logger.info(f"Received fault injection command: {fault_type}")
                self.requested_fault = fault_type
        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")

    def get_next_run(self):
        if self.requested_fault:
            fault = self.requested_fault
            self.requested_fault = None
            if fault in self.faulty_runs and self.faulty_runs[fault]:
                import random
                selected = random.choice(self.faulty_runs[fault])
                logger.info(f"Selected fault run_id={selected} for fault={fault}, mission_id={selected}, engine_id={CAN_ENGINE_ID}")
                return selected
            else:
                logger.warning(f"Requested fault {fault} not found in available runs. Continuing with a healthy run.")
                
        import random
        selected = random.choice(self.healthy_runs) if self.healthy_runs else random.choice(list(self.runs.keys()))
        logger.info(f"Selected healthy run_id={selected}, mission_id={selected}, engine_id={CAN_ENGINE_ID}")
        return selected

    def play(self):
        self.current_run_id = self.get_next_run()
        group = self.runs[self.current_run_id]
        
        period = 1.0 / self.hz
        
        for _, row in group.iterrows():
            if self.requested_fault:
                logger.info("Switching to fault mid-run as requested...")
                break
                
            telemetry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_id": CAN_ENGINE_ID,
                "mission_id": str(int(row["run_id"])),
                "mission_phase": str(row.get("mission_phase", "cruise")),
                "rpm": float(row["rpm"]),
                "cht_c": float(row["cht_c"]),
                "egt_c": float(row["egt_c"]),
                "oil_pressure_bar": float(row["oil_pressure_bar"]),
                "oil_temp_c": float(row["oil_temp_c"]),
                "fuel_flow_lph": float(row["fuel_flow_lph"]),
                "vibration_rms_g": float(row["vibration_rms_g"]),
                "battery_voltage_v": float(row["battery_voltage_v"]),
                "injection_timing_deg": float(row["injection_timing_deg"]),
                "throttle_frac": float(row["throttle_frac"]),
                "power_kw": float(row["power_kw"]),
                "ambient_temp_c": float(row["ambient_temp_c"]),
                "air_pressure_pa": float(row["air_pressure_pa"]),
                "air_density_kgm3": float(row["air_density_kgm3"]),
                "altitude_m": float(row["altitude_m"]),
            }
            
            frames = encode_telemetry(telemetry)
            for f in frames:
                msg = can.Message(arbitration_id=f.arbitration_id, data=f.data, is_extended_id=False)
                self.bus.send(msg)
                
            time.sleep(period)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/rotax_combined_clean.csv", help="Path to source CSV")
    parser.add_argument("--holdout-runs", help="Optional path to runs.json (only replays available_runs if specified)")
    parser.add_argument("--hz", type=float, default=10.0, help="Publish frequency (Hz)")
    args = parser.parse_args()

    replayer = CSVReplayer(args.data, args.holdout_runs, args.hz)
    
    try:
        while True:
            replayer.play()
    except KeyboardInterrupt:
        logger.info("Stopping replay")
    finally:
        replayer.mqtt_client.loop_stop()
        replayer.mqtt_client.disconnect()
        replayer.bus.shutdown()

if __name__ == "__main__":
    main()
