import json
import os
import glob
import csv
from PIL import Image, ImageDraw, ImageFont

IMAGE_DIR = "./images"
CELL_DIR = "./detected_cells"
REPORT_DIR = "./reports"
OUTPUT_DIR = "./diagnostic_cards"
REGISTRY_PATH = "./master_registry.csv"

START_ID = 1
END_ID = 9678

def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

def wrap_text_pixels(text, font, max_pixel_width):
    words = text.split()
    lines = []
    current_line = ""
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for word in words:
        test_line = current_line + word + " "
        try:
            w = dummy_draw.textlength(test_line, font=font)
        except AttributeError:
            w = dummy_draw.textsize(test_line, font=font)[0]
        if w <= max_pixel_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())
    return lines

def load_registry(path):
    registry = {}
    if not os.path.exists(path):
        print(f"⚠️ Warning: Registry file {path} not found. No filtering will occur.")
        return registry
    
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            registry[row['unique_id']] = row
    return registry

def generate_cards_filtered(start, end):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📂 Loading registry...")
    registry = load_registry(REGISTRY_PATH)
    
    all_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    processed_count = 0
    skipped_filter_count = 0

    for img_p in all_image_paths:
        file_id_str = os.path.basename(img_p).split('.')[0]
        try:
            file_id_int = int(file_id_str)
        except ValueError: continue

        # Range Check
        if not (start <= file_id_int <= end): continue

        if file_id_str in registry:
            reg_entry = registry[file_id_str]
            orig_label = reg_entry.get('original_label', '')
            source = reg_entry.get('source_dataset', '')

            # Condition: Skip if 'Cropped' is in label OR dataset is 'Herlev'
            if "Cropped" in orig_label or source == "Herlev":
                skipped_filter_count += 1
                continue
        
        out_p = os.path.join(OUTPUT_DIR, f"{file_id_str}_diagnostic_card.png")
        if os.path.exists(out_p): continue 
        print(f"Processing ID {file_id_str}...")
        cell_p = os.path.join(CELL_DIR, f"{file_id_str}.json")
        report_p = os.path.join(REPORT_DIR, f"{file_id_str}_report_1.json")
        if not os.path.exists(cell_p) or not os.path.exists(report_p): continue

        with open(cell_p, 'r') as f: detections = json.load(f)
        with open(report_p, 'r') as f: report_data = json.load(f)
        description = report_data.get("microscopic_description", "")

        original_img = Image.open(img_p).convert("RGB")
        w, h = original_img.size
        font_size = max(14, int(w / 60)) 
        margin = int(w / 25) 
        line_height = int(font_size * 1.6)
        font = get_font(font_size)

        draw_img = ImageDraw.Draw(original_img)
        box_thickness = max(3, int(w / 350))
        
        for cell in detections:
            try:
                coords = cell.get('box_2d', [0,0,0,0])
                if len(coords) != 4: continue
                
                ymin, xmin, ymax, xmax = coords
                
                # Calculate raw pixel values
                x0, y0 = (xmin * w) / 1000, (ymin * h) / 1000
                x1, y1 = (xmax * w) / 1000, (ymax * h) / 1000
                
                # Normalize coordinate order before drawing
                left = min(x0, x1)
                top = min(y0, y1)
                right = max(x0, x1)
                bottom = max(y0, y1)

                draw_img.rectangle([left, top, right, bottom], outline="yellow", width=box_thickness)
                
                label = cell.get('cell_type', 'Cell')
                label_y = top - (font_size + 4) if top > font_size + 10 else top + 4
                draw_img.text((left + 4, label_y), label, fill="red", font=font)
                
            except Exception as e:
                print(f"⚠️ Warning: Could not draw box for {file_id_str}: {e}")
                continue
        max_text_width = w - (2 * margin)
        wrapped_lines = wrap_text_pixels(description, font, max_text_width)
        footer_needed = (len(wrapped_lines) * line_height) + (margin * 2)

        final_card = Image.new("RGB", (w, h + footer_needed), (255, 255, 255))
        final_card.paste(original_img, (0, 0))
        draw_f = ImageDraw.Draw(final_card)
        text_y = h + margin
        for line in wrapped_lines:
            draw_f.text((margin, text_y), line, fill="black", font=font)
            text_y += line_height

        final_card.save(out_p)
        processed_count += 1
        print(f"✅ ID {file_id_str}: Card Generated.")

    print(f"\n🏁 Finished range {start}-{end}")
    print(f"📈 New Cards: {processed_count} | ⏭️ Skipped (Filter): {skipped_filter_count}")

if __name__ == "__main__":
    generate_cards_filtered(START_ID, END_ID)