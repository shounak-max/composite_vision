import os
import json
import base64
import requests
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Add your API keys here or export them as environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

# Imagenette class names mapped to their ImageNet indices
IMAGENET_CLASS_NAMES = {
    0: "tench",
    217: "English springer",
    482: "cassette player",
    491: "chain saw",
    497: "church",
    566: "French horn",
    569: "garbage truck",
    571: "gas pump",
    574: "golf ball",
    701: "parachute",
}

# Reverse mapping: lowercase name -> class index
NAME_TO_IDX = {}
for idx, name in IMAGENET_CLASS_NAMES.items():
    NAME_TO_IDX[name.lower()] = idx
    # Also add individual words for fuzzy matching
    for word in name.lower().split():
        if word not in ('a', 'an', 'the', 'of'):
            NAME_TO_IDX[word] = idx

# Additional aliases for common VLM responses
ALIASES = {
    "fish": 0, "tench fish": 0,
    "dog": 217, "springer": 217, "springer spaniel": 217, "spaniel": 217,
    "cassette": 482, "tape player": 482, "cassette deck": 482, "boombox": 482,
    "chainsaw": 491, "chain_saw": 491, "saw": 491,
    "church": 497, "cathedral": 497, "chapel": 497,
    "horn": 566, "french_horn": 566, "brass": 566,
    "truck": 569, "garbage_truck": 569, "refuse truck": 569, "dustcart": 569,
    "gas pump": 571, "gas_pump": 571, "fuel pump": 571, "petrol pump": 571, "pump": 571,
    "golf ball": 574, "golf_ball": 574, "ball": 574, "golfball": 574,
    "parachute": 701, "chute": 701,
}
NAME_TO_IDX.update(ALIASES)


def parse_vlm_response(response_text):
    """
    Parse VLM free-text response into an ImageNet class index.
    Returns the class index if a match is found, otherwise -1.
    """
    if not response_text or response_text.startswith("Error:") or response_text.startswith("Skipped"):
        return -1
    
    text = response_text.lower().strip().strip('"').strip("'").strip(".")
    
    # Direct match
    if text in NAME_TO_IDX:
        return NAME_TO_IDX[text]
    
    # Check if any class name appears in the response
    for name, idx in sorted(NAME_TO_IDX.items(), key=lambda x: -len(x[0])):
        if name in text:
            return idx
    
    return -1


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_gpt4o(image_path, prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    base64_image = encode_image(image_path)
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_gemini(image_path, prompt):
    """Evaluate an image using Google Gemini API (gemini-1.5-flash or gemini-2.0-flash)."""
    base64_image = encode_image(image_path)
    
    # Use Gemini REST API (v1beta generateContent endpoint)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 50,
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def run_vlm_evaluation(dataset_dir="dataset", results_file="results/vlm_results.json", num_samples=None):
    """
    Run VLM evaluation on the composite dataset.
    
    Args:
        dataset_dir: Path to the dataset directory containing metadata.json and images/
        results_file: Output path for the JSON results
        num_samples: Number of samples to evaluate (None = all test samples)
    """
    dataset_path = Path(dataset_dir)
    metadata_file = dataset_path / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Metadata not found at {metadata_file}")
        return
        
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Filter to test split only
    test_metadata = [item for item in metadata if item.get('split') == 'test']
    if not test_metadata:
        test_metadata = metadata  # Fallback if no split field
    
    prompts = {
        "Basic": (
            "This is a composite visual stimulus containing conflicting features "
            "(e.g., shape of one object, texture of another, or occlusion). "
            "What is the primary object class in this image? "
            "Reply with ONLY a single noun from this list: "
            "tench, English springer, cassette player, chain saw, church, "
            "French horn, garbage truck, gas pump, golf ball, parachute."
        ),
        "CoT": (
            "Analyze the features of this image step-by-step. What shape do you see? What texture do you see? "
            "What is the primary object class? Reply with your reasoning followed by the final answer as ONLY a single noun from this list: "
            "tench, English springer, cassette player, chain saw, church, "
            "French horn, garbage truck, gas pump, golf ball, parachute."
        ),
        "Adversarial": (
            "This image contains conflicting features designed to trick you. Ignore the texture and focus solely on the structural shape of the object. "
            "What is the primary object class? Reply with ONLY a single noun from this list: "
            "tench, English springer, cassette player, chain saw, church, "
            "French horn, garbage truck, gas pump, golf ball, parachute."
        )
    }
    
    # Use subset if specified, otherwise all
    subset = test_metadata[:num_samples] if num_samples else test_metadata
    
    results = []
    
    print(f"Starting VLM Evaluation on {len(subset)} composite stimuli...")
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found. GPT-4o evaluation will be skipped.")
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY/GEMINI_API_KEY not found. Gemini evaluation will be skipped.")
        
    for item in tqdm(subset, desc="Evaluating VLMs"):
        img_path = str(dataset_path / "images" / item['filename'])
        
        if not os.path.exists(img_path):
            print(f"  Skipping {item['filename']}: image not found")
            continue
        
        item_results = {
            "filename": item['filename'],
            "class1": item['class1'],
            "class2": item['class2'],
            "composition_type": item['composition_type'],
            "salience": item['salience'],
        }
        
        for prompt_name, prompt_text in prompts.items():
            gpt4o_raw = evaluate_gpt4o(img_path, prompt_text) if OPENAI_API_KEY else "Skipped - Missing Key"
            gemini_raw = evaluate_gemini(img_path, prompt_text) if GOOGLE_API_KEY else "Skipped - Missing Key"
            
            item_results[f"gpt4o_raw_{prompt_name}"] = gpt4o_raw
            item_results[f"gpt4o_pred_{prompt_name}"] = parse_vlm_response(gpt4o_raw)
            item_results[f"gemini_raw_{prompt_name}"] = gemini_raw
            item_results[f"gemini_pred_{prompt_name}"] = parse_vlm_response(gemini_raw)
            
        results.append(item_results)
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # Compute accuracy and print summary
    print(f"\nVLM Evaluation complete! Results saved to {results_file}")
    _print_vlm_summary(results)
    
    # Integrate results into benchmark_results.csv
    _integrate_vlm_results(results, os.path.dirname(results_file))


def _print_vlm_summary(results):
    """Print accuracy summary for VLM results."""
    prompt_names = ["Basic", "CoT", "Adversarial"]
    for model_base in ["gpt4o", "gemini"]:
        for prompt_name in prompt_names:
            pred_key = f"{model_base}_pred_{prompt_name}"
            preds = [r[pred_key] for r in results if r.get(pred_key, -1) != -1]
            model_key = f"{model_base.upper()}_{prompt_name}"
            if not preds:
                print(f"  {model_key}: No valid predictions")
                continue
            
            correct = sum(1 for r in results if r.get(pred_key, -1) == r['class1'])
            total = len(results)
            valid = len(preds)
            
            print(f"  {model_key}: {correct}/{total} correct ({100*correct/total:.2f}% accuracy), "
                  f"{valid}/{total} parsed successfully ({100*valid/total:.1f}%)")


def _integrate_vlm_results(results, results_dir):
    """Add VLM results as rows in benchmark_results.csv for unified analysis."""
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    
    vlm_rows = []
    prompt_names = ["Basic", "CoT", "Adversarial"]
    for r in results:
        for model_base, display_name in [("gpt4o", "GPT-4o"), ("gemini", "Gemini")]:
            for prompt_name in prompt_names:
                pred_key = f"{model_base}_pred_{prompt_name}"
                model_name = f"{display_name}_{prompt_name}"
                if r.get(pred_key, -1) == -1:
                    continue
                vlm_rows.append({
                    'filename': r['filename'],
                    'class1': r['class1'],
                    'class2': r['class2'],
                    'composition_type': r['composition_type'],
                    'salience': r['salience'],
                    'model': model_name,
                    'top1_pred': r[pred_key],
                    'top1_conf': -1.0,  # VLMs don't return confidence scores
                    'top5_preds': str([r[pred_key]]),
                    'inference_time': -1.0,  # API latency not comparable
                })
    
    if not vlm_rows:
        print("Warning: No valid VLM predictions to integrate.")
        return
    
    df_vlm = pd.DataFrame(vlm_rows)
    
    if os.path.exists(csv_path):
        df_base = pd.read_csv(csv_path)
        # Remove any existing VLM rows to avoid duplicates
        vlm_model_names = [f"{m}_{p}" for m in ["GPT-4o", "Gemini"] for p in prompt_names]
        df_base = df_base[~df_base['model'].isin(vlm_model_names)]
        df_combined = pd.concat([df_base, df_vlm], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"Integrated {len(vlm_rows)} VLM predictions into {csv_path}")
    else:
        df_vlm.to_csv(csv_path, index=False)
        print(f"Created {csv_path} with {len(vlm_rows)} VLM predictions")


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_vlm_evaluation(
        dataset_dir=os.path.join(base_dir, "dataset"),
        results_file=os.path.join(base_dir, "results", "vlm_results.json"),
        num_samples=None  # Evaluate ALL test samples
    )
