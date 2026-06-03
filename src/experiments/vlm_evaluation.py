import os
import json
import base64
import requests
from tqdm import tqdm
from pathlib import Path

# Add your API keys here or export them as environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    return f"Error: {response.status_code}"

def evaluate_claude(image_path, prompt):
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    base64_image = encode_image(image_path)
    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 50,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['content'][0]['text'].strip()
    return f"Error: {response.status_code}"

def run_vlm_evaluation(dataset_dir="dataset", results_file="results/vlm_results.json", num_samples=10):
    dataset_path = Path(dataset_dir)
    metadata_file = dataset_path / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Metadata not found at {metadata_file}")
        return
        
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    prompt = "This is a composite visual stimulus containing conflicting features (e.g., shape of one object, texture of another, or occlusion). What is the primary object class in this image? Reply with a single noun corresponding to an ImageNet category."
    
    results = []
    
    # Evaluate a small subset for demonstration (or change to len(metadata))
    subset = metadata[:num_samples]
    
    print(f"Starting VLM Evaluation on {len(subset)} composite stimuli...")
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found. GPT-4o evaluation will be skipped.")
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not found. Claude 3.5 evaluation will be skipped.")
        
    for item in tqdm(subset, desc="Evaluating VLMs"):
        img_path = str(dataset_path / item['filename'])
        
        gpt4o_pred = evaluate_gpt4o(img_path, prompt) if OPENAI_API_KEY else "Skipped - Missing Key"
        claude_pred = evaluate_claude(img_path, prompt) if ANTHROPIC_API_KEY else "Skipped - Missing Key"
        
        results.append({
            "filename": item['filename'],
            "class1": item['class1'],
            "class2": item['class2'],
            "composition_type": item['composition_type'],
            "salience": item['salience'],
            "gpt4o_pred": gpt4o_pred,
            "claude_pred": claude_pred
        })
        
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nVLM Evaluation complete! Results saved to {results_file}")

if __name__ == "__main__":
    run_vlm_evaluation()
