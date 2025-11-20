
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import io
from PIL import Image
import requests
import re
import os
from fastapi.middleware.cors import CORSMiddleware
import threading
import time

app = FastAPI()

# Load the pre-trained ResNet50 model
model = ResNet50(weights="imagenet")
# Prepare models for intermediate outputs
layer_names = [
    'conv1_conv',
    'pool1_pool',
    'conv2_block1_out',
    'avg_pool',
    'predictions'
]
intermediate_models = {name: Model(inputs=model.input, outputs=model.get_layer(name).output) for name in layer_names}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

def fetch_wikidata_facts(query):
    # Step 1: Search for the entity
    search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json"
    try:
        r = requests.get(search_url)
        if r.status_code == 200:
            data = r.json()
            if data.get("search"):
                entity_id = data["search"][0]["id"]
                # Step 2: Fetch entity data
                entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                r2 = requests.get(entity_url)
                if r2.status_code == 200:
                    entity_data = r2.json()
                    entity = entity_data["entities"][entity_id]
                    claims = entity.get("claims", {})
                    labels = entity.get("labels", {})
                    descriptions = entity.get("descriptions", {})
                    facts = {}
                    # Label and description
                    facts["wikidata_label"] = labels.get("en", {}).get("value")
                    facts["wikidata_description"] = descriptions.get("en", {}).get("value")
                    # Inventor (P61)
                    if "P61" in claims:
                        inventor = claims["P61"][0]["mainsnak"]["datavalue"]["value"]
                        if isinstance(inventor, dict) and "id" in inventor:
                            inventor_id = inventor["id"]
                            inventor_url = f"https://www.wikidata.org/wiki/Special:EntityData/{inventor_id}.json"
                            r3 = requests.get(inventor_url)
                            if r3.status_code == 200:
                                inventor_data = r3.json()
                                inventor_entity = inventor_data["entities"][inventor_id]
                                inventor_label = inventor_entity.get("labels", {}).get("en", {}).get("value")
                                facts["inventor"] = inventor_label
                    # Use (P366)
                    if "P366" in claims:
                        use = claims["P366"][0]["mainsnak"]["datavalue"]["value"]
                        if isinstance(use, dict) and "id" in use:
                            use_id = use["id"]
                            use_url = f"https://www.wikidata.org/wiki/Special:EntityData/{use_id}.json"
                            r4 = requests.get(use_url)
                            if r4.status_code == 200:
                                use_data = r4.json()
                                use_entity = use_data["entities"][use_id]
                                use_label = use_entity.get("labels", {}).get("en", {}).get("value")
                                facts["use"] = use_label
                    # Instance of (P31)
                    if "P31" in claims:
                        instance = claims["P31"][0]["mainsnak"]["datavalue"]["value"]
                        if isinstance(instance, dict) and "id" in instance:
                            instance_id = instance["id"]
                            instance_url = f"https://www.wikidata.org/wiki/Special:EntityData/{instance_id}.json"
                            r5 = requests.get(instance_url)
                            if r5.status_code == 200:
                                instance_data = r5.json()
                                instance_entity = instance_data["entities"][instance_id]
                                instance_label = instance_entity.get("labels", {}).get("en", {}).get("value")
                                facts["instance_of"] = instance_label
                    # Image (P18)
                    if "P18" in claims:
                        image_name = claims["P18"][0]["mainsnak"]["datavalue"]["value"]
                        # Convert Wikimedia image name to URL
                        commons_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{re.sub(' ', '_', image_name)}"
                        facts["wikidata_image"] = commons_url
                    return facts
    except Exception as e:
        return {"wikidata_error": str(e)}
    return {}

def fetch_duckduckgo_info(query):
    ddg_url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        r = requests.get(ddg_url)
        if r.status_code == 200:
            data = r.json()
            # Extract related topics as a list of strings
            related = []
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict):
                    if "Text" in topic:
                        related.append(topic["Text"])
                    # Some topics have a 'Topics' list
                    if "Topics" in topic:
                        for subtopic in topic["Topics"]:
                            if "Text" in subtopic:
                                related.append(subtopic["Text"])
            return {
                "summary": data.get("AbstractText"),
                "image": data.get("Image"),
                "url": data.get("AbstractURL"),
                "heading": data.get("Heading"),
                "related": related,
                "description": data.get("Description"),
                "type": data.get("Type"),
            }
    except Exception as e:
        return {"error": str(e)}
    return {}

def fetch_wikipedia_info(query):
    wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    try:
        r = requests.get(wiki_url)
        if r.status_code == 200:
            data = r.json()
            return {
                "summary": data.get("extract"),
                "image": data.get("thumbnail", {}).get("source"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
                "heading": data.get("title"),
                "description": data.get("description"),
                "type": data.get("type"),
            }
    except Exception as e:
        return {"error": str(e)}
    return {}

def fetch_wikipedia_sections(query):
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    r = requests.get(summary_url)
    if r.status_code != 200:
        return {}
    data = r.json()
    title = data.get("title", query)
    sections_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"
    r = requests.get(sections_url)
    if r.status_code != 200:
        return {}
    data = r.json()
    sections = data.get("remaining", {}).get("sections", [])
    info = {}
    for section in sections:
        heading = section.get("line", "").lower()
        text = section.get("text", "")
        if "history" in heading:
            info["history"] = text
        if "use" in heading or "application" in heading:
            info["uses"] = text
        if "invent" in heading or "origin" in heading or "creator" in heading:
            info["inventor"] = text
    return info

def fetch_wikipedia_info_with_context(query, context_list=None):
    # Try the original query
    info = fetch_wikipedia_info(query)
    # If ambiguous, try with context
    if info.get("type") == "disambiguation" and context_list:
        for context in context_list:
            refined_query = f"{query}_{context}".replace(' ', '_')
            refined_info = fetch_wikipedia_info(refined_query)
            if refined_info.get("summary") and refined_info.get("type") != "disambiguation":
                return refined_info
    return info

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    def log_cnn_process(filename, img_shape, resized_shape, norm_example, decoded):
        print(f"\n[CNN PROCESS LOG]")
        print(f"Image Upload: {filename} ({img_shape[0]}x{img_shape[1]} RGB)")
        print(f"Load Image: shape {img_shape[0]}, {img_shape[1]}, 3")
        print(f"Resize to {resized_shape[0]}x{resized_shape[1]} (Standard CNN Input Size)")
        print(f"Normalize Pixels: Divide by 255, Example: {norm_example[0]} → {norm_example[1]:.3f}")
        print(f"ResNet Preprocessing: Mean Subtraction, Scaling")
        print(f"ResNet Block 1: Convolution + BatchNorm + ReLU")
        print(f"ResNet Block 2: Convolution + BatchNorm + ReLU")
        print(f"ResNet Block 3: Convolution + BatchNorm + ReLU")
        print(f"Global Average Pooling: Reduce to 2048 Features")
        print(f"Fully Connected Layer: Dense(1000) for ImageNet")
        print("Classification Result (Top Predictions):")
        for pred in decoded:
            print(f"  {pred[1]}: {pred[2]*100:.2f}%")
        print(f"Return to System: Category: {decoded[0][1]}, Confidence: {decoded[0][2]*100:.2f}%\n")

    # Read image file
    contents = await file.read()

    print("\n[PROCESSING STAGE]")
    print("Step 1: Image Upload and Preprocessing")
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    print(f"Uploaded image: {file.filename}, shape: {img.size[1]}, {img.size[0]}, 3 (RGB)")
    img_shape = (img.size[1], img.size[0])
    img = img.resize((224, 224))
    print(f"Resized to: 224x224 (Standard CNN input size)")
    resized_shape = (224, 224)
    x = image.img_to_array(img)
    print(f"Converted to array, shape: {x.shape}")
    x_norm = x / 255.0
    norm_example = (int(x[0,0,0]), x_norm[0,0,0])
    print(f"Normalized pixels: Example {norm_example[0]} → {norm_example[1]:.3f}")
    x = np.expand_dims(x, axis=0)
    print(f"Expanded dims for batch, shape: {x.shape}")
    x = preprocess_input(x)
    print(f"Preprocessed for ResNet50 (mean subtraction, scaling)")

    print("\n[LAYER COMPUTATION]")
    conv1_out = intermediate_models['conv1_conv'].predict(x)
    print(f"conv1_conv output shape: {conv1_out.shape}")
    print(f"conv1_conv sample values: {conv1_out[0, :2, :2, :4]}")

    pool1_out = intermediate_models['pool1_pool'].predict(x)
    print(f"pool1_pool output shape: {pool1_out.shape}")
    print(f"pool1_pool sample values: {pool1_out[0, :2, :2, :4]}")

    block1_out = intermediate_models['conv2_block1_out'].predict(x)
    print(f"conv2_block1_out output shape: {block1_out.shape}")
    print(f"conv2_block1_out sample values: {block1_out[0, :2, :2, :4]}")

    avgpool_out = intermediate_models['avg_pool'].predict(x)
    print(f"avg_pool output shape: {avgpool_out.shape}")
    print(f"avg_pool sample values: {avgpool_out[0, :4]}")

    logits = intermediate_models['predictions'].predict(x)
    print(f"predictions (logits) output shape: {logits.shape}")
    print(f"predictions sample logits: {logits[0, :5]}")

    print("\n[VISION TRANSFORMER ATTENTION MECHANISM]")
    print("(Note: Vision Transformer not implemented in this pipeline. Placeholder for future extension.)")

    print("\n[CLASSIFICATION OUTPUT CALCULATION]")
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    print(f"Softmax sample probabilities: {softmax[0, :5]}")
    print(f"Softmax sample probabilities: {softmax[0, :5]}")

    preds = logits
    decoded = decode_predictions(preds, top=3)[0]
    print("Top 3 predictions:")
    for i, pred in enumerate(decoded):
        print(f"    Class: {pred[1]}, Confidence: {pred[2]:.4f}")
    print("\n[EXPLANATION: HOW PREDICTIONS ARE MADE]")
    print("1. The model outputs 1000 logits (raw scores), one for each ImageNet class.")
    print("2. These logits are converted to probabilities using the softmax function:")
    print("   softmax[i] = exp(logit[i]) / sum(exp(logit[j]) for all j)")
    print("3. The top 3 probabilities are selected and mapped to class names using the ImageNet label index.")
    print("4. For your image, the highest probabilities correspond to the classes:")
    for i, pred in enumerate(decoded):
        print(f"   {i+1}. {pred[1]} (probability: {pred[2]*100:.2f}%)")
    print("5. The class names (e.g., 'laptop', 'notebook', 'space_bar') are defined by the ImageNet dataset and may reflect visual similarity, not literal object type.")
    print("6. The model does not 'know' the real-world context, only what it has learned from millions of labeled images.")
    predictions = [
        {"className": pred[1], "confidence": float(pred[2])}
        for pred in decoded
    ]

    print("\n[COMPLETE PROCESSING PIPELINE]")
    print("Stage 1: Preprocessing → Stage 2: Layer Computation → Stage 3: (Vision Transformer) → Stage 4: Classification Output → Stage 5: Info Retrieval")
    log_cnn_process(file.filename, img_shape, resized_shape, norm_example, decoded)

    top_class = predictions[0]["className"].replace('_', ' ')
    ddg_query = f"what is a {top_class}"
    ddg_info = fetch_duckduckgo_info(ddg_query)
    context_list = ["computer", "device", "tool", "object", "technology"]
    if not ddg_info.get("summary"):
        wiki_query = top_class.replace(' ', '_')
        wiki_info = fetch_wikipedia_info_with_context(wiki_query, context_list)
        wiki_sections = fetch_wikipedia_sections(wiki_query)
        wikidata_facts = fetch_wikidata_facts(top_class)
        info = {**wiki_info, **wiki_sections, **wikidata_facts} if wiki_info.get("summary") else {**ddg_info, **wikidata_facts}
    else:
        wiki_query = top_class.replace(' ', '_')
        wiki_sections = fetch_wikipedia_sections(wiki_query)
        wikidata_facts = fetch_wikidata_facts(top_class)
        info = {**ddg_info, **wiki_sections, **wikidata_facts}
    if info.get("type") == "disambiguation":
        info["ambiguous"] = "True"
        # Print summary to terminal
        print("\n[CNN FINAL RESULT]")
        print(f"Top Prediction: {top_class} ({predictions[0]['confidence']*100:.2f}%)")
        print("Info Summary:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("[END OF RESULT]\n")
    return JSONResponse({"predictions": predictions, "info": info})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)