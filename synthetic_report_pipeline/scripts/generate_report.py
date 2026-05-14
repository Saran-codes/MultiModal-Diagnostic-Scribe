import os
import json
import pandas as pd
import PIL.Image
from pathlib import Path
from google.genai import Client, types
import signal
import sys
import time
import random

WORKER_ID = int(sys.argv[1]) 
TOTAL_WORKERS = int(sys.argv[2])

stop_requested = False

def signal_handler(sig, frame):
    global stop_requested
    print("\n🛑 Stop request received! Finishing current image and saving state before exiting...")
    stop_requested = True


def interruptible_sleep(seconds):
    """Sleeps for the duration but checks for stop_requested every second."""
    for _ in range(seconds):
        if stop_requested:
            break
        time.sleep(1)

signal.signal(signal.SIGINT, signal_handler)

LIMIT = 25  # max images per session
processed_count = 0

FAILURE_THRESHOLD = 50
failure_count = 0
failed_ids = []

model_3_p = "gemini-3-pro-preview"
model_3_f = "gemini-3-flash-preview"
model_2_5_p = "gemini-2.5-pro"
model_2_5_f = "gemini-2.5-flash"

base_path = Path("./")
img_dir = base_path / "images"
out_rep_dir = base_path / "reports"
out_cells_dir = base_path / "detected_cells"
registry_path = base_path / "master_registry.csv"
status_path = base_path / "project_status.json"
log_path = base_path / "processing_log.txt"
meta_stage_dir = base_path / "staging_area"

client = Client()
df = pd.read_csv(registry_path)

# Ensure ID column is treated as strings for zfill consistency
df['unique_id'] = df['unique_id'].astype(str).str.zfill(6)

def log_event(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] Worker {WORKER_ID}: {message}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg)
    except IOError:
        time.sleep(0.1)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg)


# with open(status_path, 'r') as f:
#     status = json.load(f)

# def save_state(df_to_save, status_to_save):
#     """Saves progress to both Registry and Status JSON."""
#     df_to_save.to_csv(registry_path, index=False)
#     with open(status_path, 'w') as f:
#         json.dump(status_to_save, f, indent=4)

SYSTEM_INSTRUCTION = """
You are an expert cytopathologist specialized in cervical cytology. 
Analyze the provided Pap smear images which may range from single isolated cells to complex multi-celled clusters.
You are also provided with a label/caption for each image. For collages, analyze sub-panels independently and use 'and' to join differing diagnoses in the interpretation field in same report.
Categories include: NILM, LSIL, HSIL, Squamous Cell Carcinoma and Adeno Carcinoma.

Operational Protocol:
1. Spatial Analysis: Identify and localize specific diagnostic cells.
2. Multi-Level Morphological Review: 
   - Cellular Level: Evaluate N/C ratio, nuclear membrane contour, chromatin distribution (coarse vs. fine), and cytoplasmic quality.
   - Structural Level: Assess three-dimensional clusters, cell-in-cell patterns, and background elements (e.g., inflammation, diathesis).
3. Dual Reporting: Generate TWO STRICTLY independent professional Bethesda reports for each image.

GROUNDING RULE: Do not infer or "hallucinate" features not clearly visible. If an image is blurry or a nucleus is obscured by debris, explicitly state "Unsatisfactory for evaluation". Before finalizing each report, cross-check: "Does the microscopic description provide sufficient evidence for the interpretation?".

Requirements:
- Use strictly standard medical terminology (Bethesda System). No Conversational filler language.
- Provide an extremely detailed 'microscopic_description' for each report, explaining the visual evidence for the interpretation.
- Return strictly in JSON format.
"""

response_schema = {
    "type": "OBJECT",
    "properties": {
        "detected_cells": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "cell_type": {"type": "STRING"},
                    "box_2d": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                    "features": {"type": "STRING"}
                }
            }
        },
        "bethesda_report_1": {
            "type": "OBJECT",
            "properties": {
                "specimen_type": {"type": "STRING"},
                "specimen_adequacy": {"type": "STRING"},
                "general_categorization": {"type": "STRING"},
                "microscopic_description": {"type": "STRING"},
                "interpretation": {"type": "STRING"},
                "recommendation": {"type": "STRING"}
            },
            "required": ["specimen_adequacy", "microscopic_description", "interpretation"]
        },
        "bethesda_report_2": {
            "type": "OBJECT",
            "properties": {
                "specimen_type": {"type": "STRING"},
                "specimen_adequacy": {"type": "STRING"},
                "general_categorization": {"type": "STRING"},
                "microscopic_description": {"type": "STRING"},
                "interpretation": {"type": "STRING"},
                "recommendation": {"type": "STRING"}
            },
            "required": ["specimen_adequacy", "microscopic_description", "interpretation"]
        }
    },
    "required": ["detected_cells", "bethesda_report_1", "bethesda_report_2"]
}

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    )
]

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    safety_settings=safety_settings,
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_level=types.ThinkingLevel.HIGH
    ),
    temperature=0.3,
    response_mime_type="application/json",
    response_schema=response_schema
)

pending_images = df[df['report_generated'] == False]

print(f"👷 Worker {WORKER_ID}/{TOTAL_WORKERS} active. Batch Limit: {LIMIT}")


for index, row in pending_images.iterrows():
    
    if stop_requested:
        break
    
    if index % TOTAL_WORKERS != WORKER_ID:
        continue
    
    if processed_count >= LIMIT:
        print(f"\nReached session limit of {LIMIT}. Stopping.")
        break
    
    if failure_count > FAILURE_THRESHOLD:
        msg = f"\nSTOPPING: Failure threshold ({FAILURE_THRESHOLD}) reached. Failed IDs: {failed_ids}"
        print(msg)
        log_event(msg)
        break
    
    img_id = row['unique_id']
    img_path = img_dir / f"{img_id}.png"
    
    if not img_path.exists():
        failure_count += 1
        failed_ids.append(img_id)
        print(f"Error: Image {img_id}.png not found. Skipping.")
        continue
    
    if (meta_stage_dir / f"meta_{img_id}.json").exists():
        continue
    
    prompt = f"Analyze this Pap smear. Context: Label is {row['mapped_label']} (Other details: {row['original_label']})."

    print(f"[[Worker {WORKER_ID}]][{processed_count + 1}/{LIMIT}] Processing {img_id}...")

    try:
        img = PIL.Image.open(img_path)
        
        api_start = time.time()
        response = client.models.generate_content(
            model= model_3_f,
            contents=[img, prompt],
            config=config
        )
        api_end = time.time()
        api_duration = api_end - api_start

        result = json.loads(response.text)
        usage = response.usage_metadata
        t_tokens = int(usage.total_token_count - usage.prompt_token_count - usage.candidates_token_count)
        
        receipt = {
            "unique_id": img_id,
            "input_tokens": int(usage.prompt_token_count),
            "output_tokens": int(usage.candidates_token_count),
            "thinking_tokens": t_tokens
        }
        
        with open(meta_stage_dir / f"meta_{img_id}.json", "w") as f:
            json.dump(receipt, f, indent=4)

        with open(out_rep_dir / f"{img_id}_report_1.json", "w") as f:
            json.dump(result['bethesda_report_1'], f, indent=4)
        with open(out_rep_dir / f"{img_id}_report_2.json", "w") as f:
            json.dump(result['bethesda_report_2'], f, indent=4)
        with open(out_cells_dir / f"{img_id}.json", "w") as f:
            json.dump(result.get('detected_cells', []), f, indent=4)

        # # 4. Update Registry Dataframe
        # df.at[index, 'report_generated'] = True
        # df.at[index, 'num_reports'] = 2
        # # Track thinking tokens if available in metadata
        # df.at[index, 'thinking_tokens'] = (usage.total_token_count - usage.prompt_token_count - usage.candidates_token_count)

        # # 5. Update Status JSON
        # status['total_reports_generated'] += 1
        # status['input_tokens_used'] += int(usage.prompt_token_count)
        # status['output_tokens_used'] += int(usage.candidates_token_count)
        # status['thinking_tokens_used'] += int(df.at[index, 'thinking_tokens'])

        # # 6. Periodic State Save (After every successful image)
        # save_start = time.time()
        # save_state(df, status)
        # save_end = time.time()
        # save_duration = save_end - save_start
        
        processed_count += 1
        print(f"✅ Worker {WORKER_ID} | Done {img_id} | API: {api_duration:.2f}s")
        jitter_sleep = random.randint(2, 6)
        interruptible_sleep(jitter_sleep)

    except Exception as e:
        failure_count += 1
        failed_ids.append(img_id)
        error_msg = f"❌ Failed {img_id}: {e}"
        print(f"Worker {WORKER_ID} | {error_msg}")
        log_event(error_msg)
        if stop_requested:
            break
        
        jitter_sleep = random.randint(15, 60)
        
        interruptible_sleep(jitter_sleep) 
        continue

client.close()
print(f"Session Finished. Total processed: {processed_count}")
if stop_requested:
    sys.exit(0)