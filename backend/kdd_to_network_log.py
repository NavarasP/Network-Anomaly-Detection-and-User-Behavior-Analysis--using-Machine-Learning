#!/usr/bin/env python3
"""
NSL-KDD to network.log Converter

Reads NSL-KDD dataset (KDDTest+.txt) and converts each line to the network.log CSV format
that the backend expects, then appends it to network.log in real-time.

NSL-KDD format has 41 features + label + difficulty. We'll map the relevant features
to our expected format and synthesize user behavior features.

Usage:
    python kdd_to_network_log.py --delay 1.0 --limit 100
    
    --delay: seconds between appending each line (default: 1.0, simulates real-time)
    --limit: max number of lines to process (default: all)
    --start: line number to start from (default: 0)
"""
import argparse
import time
import random
import os
from datetime import datetime

# Paths
KDD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "nsl-kdd", "KDDTest+.txt")
LOG_FILE = os.path.join(os.path.dirname(__file__), "network.log")

# NSL-KDD feature indices (0-based)
# Format: duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,...
# We need: duration, src_bytes, dst_bytes, wrong_fragment, urgent, count_same_dst, srv_count, protocol_type, service, flag
# Plus synthesized user features: login_hour, avg_login_hour, device_count, new_device_flag, sensitive_access

def parse_kdd_line(line):
    """Parse a KDD line and write all 41 features to network.log.
    
    KDD format: 41 features + attack_type + difficulty
    We write: timestamp, attack_type, then all 41 features as-is (no mapping)
    """
    try:
        parts = line.strip().split(',')
        if len(parts) < 42:
            return None
        
        # Extract all 41 features (indices 0-40)
        features = parts[0:41]
        attack_type = parts[41] if len(parts) > 41 else "normal"
        
        # Add timestamp and attack type for reference
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Build CSV: timestamp, attack_type, then all 41 KDD features
        log_line = f"{timestamp},{attack_type}," + ",".join(features)
        
        return log_line, attack_type
    
    except Exception as e:
        print(f"Failed to parse KDD line: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Convert NSL-KDD to network.log in real-time")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between log lines")
    parser.add_argument("--limit", type=int, default=None, help="Max number of lines to process")
    parser.add_argument("--start", type=int, default=0, help="Line number to start from (0-based)")
    parser.add_argument("--burst", action="store_true", help="No delay, append all lines at once")
    args = parser.parse_args()
    
    if not os.path.exists(KDD_FILE):
        print(f"❌ KDD file not found: {KDD_FILE}")
        return
    
    print(f"📂 Reading from: {KDD_FILE}")
    print(f"📝 Writing to: {LOG_FILE}")
    print(f"⏱️  Delay: {args.delay}s (burst: {args.burst})")
    print(f"🔢 Start line: {args.start}, Limit: {args.limit or 'all'}")
    print()
    
    with open(KDD_FILE, 'r') as kdd:
        lines = kdd.readlines()
    
    total = len(lines)
    start = args.start
    end = min(start + args.limit, total) if args.limit else total
    
    print(f"Processing lines {start} to {end-1} (total: {total})")
    
    processed = 0
    anomalies = 0
    
    for i in range(start, end):
        line = lines[i]
        log_line, attack_type = parse_kdd_line(line)
        
        if log_line:
            with open(LOG_FILE, 'a') as f:
                f.write(log_line + "\n")
            
            is_anomaly = attack_type != "normal"
            if is_anomaly:
                anomalies += 1
            
            processed += 1
            status = "🚨" if is_anomaly else "✅"
            print(f"{status} [{processed}/{end-start}] {attack_type:20s} | {log_line[:80]}...")
            
            if not args.burst and i < end - 1:
                time.sleep(args.delay)
    
    print()
    print(f"✅ Done! Processed {processed} lines ({anomalies} anomalies, {processed-anomalies} normal)")
    print(f"📝 Appended to {LOG_FILE}")


if __name__ == "__main__":
    main()
