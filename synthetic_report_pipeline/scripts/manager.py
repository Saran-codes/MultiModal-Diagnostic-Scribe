import subprocess
import time
import sys
import signal
import math
import pandas as pd
from sync import run_sync
from pathlib import Path
import time

NUM_WORKERS = 3
TOTAL_IMAGES_TO_PROCESS = 1000
WORKER_INTERNAL_LIMIT = 25  # must match LIMIT in generate_report.py

base_path = Path("./")
registry_path = base_path / "master_registry.csv"

def interruptible_sleep(seconds):
    """Sleeps for the duration but checks for stop_requested every second."""
    for _ in range(seconds):
        if stop_requested:
            break
        time.sleep(1)

def check_remaining_work():
    if not registry_path.exists():
        print("❌ Error: master_registry.csv not found!")
        sys.exit(1)
        
    df_check = pd.read_csv(registry_path)
    total_rows = len(df_check)
    completed = len(df_check[df_check['report_generated'] == True])
    remaining = total_rows - completed
    
    print("="*40)
    print(f"📊 PROJECT STATUS AT LAUNCH")
    print(f"Total Dataset:    {total_rows}")
    print(f"Already Done:     {completed} ({(completed/total_rows)*100:.1f}%)")
    print(f"Pending Images:   {remaining}")
    print("="*40)
    return remaining

# Run the check
remaining_work = check_remaining_work()

TOTAL_IMAGES_TO_PROCESS = min(TOTAL_IMAGES_TO_PROCESS, remaining_work+2000)

# Calculate how many loops we need
IMAGES_PER_LOOP = NUM_WORKERS * WORKER_INTERNAL_LIMIT
NUM_LOOPS = math.ceil(TOTAL_IMAGES_TO_PROCESS / IMAGES_PER_LOOP)

active_processes = []
stop_requested = False



def manager_signal_handler(sig, frame):
    global stop_requested
    print("\n🛑 Manager: Shutdown requested! Signaling workers to finish current image...")
    stop_requested = True
    # Send Ctrl+C to all currently running workers
    for p in active_processes:
        if p.poll() is None: # If process is still running
            p.send_signal(signal.SIGINT)

# Register the interrupt handler
signal.signal(signal.SIGINT, manager_signal_handler)

print(f"🚀 Manager Initialized")
print(f"Goal: {TOTAL_IMAGES_TO_PROCESS} images | Workers: {NUM_WORKERS} | Batch Size: {IMAGES_PER_LOOP}")
print(f"Total planned loops: {NUM_LOOPS}")

start_manager_time = time.time()

for loop in range(1, NUM_LOOPS + 1):
    if stop_requested:
        break

    print(f"\n--- 🔄 LOOP {loop}/{NUM_LOOPS} STARTING ---")
    active_processes = []

    for i in range(NUM_WORKERS):
        interruptible_sleep(2*i+1)
        p = subprocess.Popen([
            sys.executable, "scripts/generate_report.py", str(i), str(NUM_WORKERS)
        ])
        active_processes.append(p)

    for p in active_processes:
        p.wait()

    # Run sync even on stop to capture the last batch's receipts
    print(f"📥 Loop {loop} workers finished. Running Accountant...")
    run_sync()
    
    elapsed_total = time.time() - start_manager_time # Define start_manager_time at top
    avg_time_per_loop = elapsed_total / loop
    remaining_loops = NUM_LOOPS - loop
    est_remaining_seconds = remaining_loops * avg_time_per_loop
    
    rem_hours = int(est_remaining_seconds // 3600)
    rem_mins = int((est_remaining_seconds % 3600) // 60)
    
    finish_timestamp = time.ctime(time.time() + est_remaining_seconds)
    
    print(f"⏱️ Loop speed: {avg_time_per_loop/60:.2f} min")
    print(f"⏳ Remaining: {rem_hours}h {rem_mins}m | Est. Finish: {finish_timestamp}")

    if stop_requested:
        print("Manager: Graceful exit complete.")
        break

print("\n🏁 PROCESS FINISHED. Check master_registry.csv for final results.")