import os
import base64
import json
import csv
import time
import requests
import argparse
import logging
import concurrent.futures
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Any, Tuple, Set
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openrouter_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("openrouter_bencher")

# Attempt to import tiktoken, but make it optional for the actual run
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken library not found. Dry run token estimation for GPT-like models will be less accurate.")

# --- 1. CONFIGURATION ---
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"  # <--- !!! REPLACE THIS !!!
HTTP_REFERER = ""  # Optional, e.g., "https://myproject.ai"
X_TITLE = "VLM Grasping Study"  # Optional, e.g., "VLM Grasping Study"

IMAGE_DIRECTORY = "./photo_booth_images/new"  # <--- !!! SET THIS PATH !!!
OUTPUT_CSV_FILE = "vlm_benchmark_results-may25.csv"
CHECKPOINT_FILE = "benchmark_checkpoint-may25.json"
NUMBER_OF_RETRIES = 1
MAX_WORKERS = 10  # Maximum number of concurrent threads
RATE_LIMIT_BACKOFF_BASE = 2  # Base for exponential backoff
RATE_LIMIT_MAX_RETRIES = 5  # Maximum number of retries for rate limit errors

# CSV headers for results - defined centrally to ensure consistency
CSV_HEADERS = [
    "Image_Name", "Image_Path", "VLM_Model", "Retry_Attempt", "Prompt_Type",
    "Prompt_Text", "Raw_Response",
    "Parsed_Object_Name", "Parsed_Object_Shape", "Parsed_Object_Dimensions_mm", "Parsed_Object_Orientation",
    "Parsed_Grasp_Type", "Parsed_Hand_Rotation_Deg", "Parsed_Hand_Aperture_mm", "Parsed_Num_Fingers",
    "JSON_Valid", "Required_Keys_Present",
    "Error_Message", "Latency_Seconds", "Cost_USD",
    "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"
]

# Models to test. Ensure these IDs are correct for OpenRouter.
MODELS_TO_TEST = [
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4.1",
    "google/gemma-3-27b-it",
    "google/gemini-2.5-flash-preview",
    "mistralai/mistral-medium-3",
    "google/gemini-2.5-flash-preview-05-20"
    "anthropic/claude-sonnet-4"
]

ALL_MODEL_DETAILS = {}

# If you want to ask individual prompts, you can define them here.
# INDIVIDUAL_PROMPTS = [
#     {
#         "name": "simple_text",
#         "text": "Describe the object in the image."
#     },
# ]

INDIVIDUAL_PROMPTS = []  # No individual prompts defined for this benchmark

COMPLEX_PROMPT_TEXT = """
You are an expert roboticist controlling a LEFT bionic hand with a camera on top. You will see ONE image.
Start pose: palm down, thumb pointing right.

Task
----
Identify the main object in the image and derive its properties and suitable grasp parameters for the LEFT bionic hand.

Return **EXACTLY one line of JSON**—no commentary, no line breaks:
{"object_name":"<object_name>","object_shape":"<cuboid|cylinder|sphere>","object_dimensions_mm":"<W>x<H>x<D>","object_orientation":"<horizontal|vertical>","grasp_type":<0|1>,"hand_rotation_deg":<float>,"hand_aperture_mm":<float>,"num_fingers":<int>}

Parameter Rules and Clarifications
----------------------------------

**Object Properties:**
• `object_name`: Generic name of the main identified object (e.g., "bottle", "book", "screwdriver").
• `object_shape`: The primary geometric shape of the main object. Choose from: "cuboid", "cylinder", "sphere".
• `object_dimensions_mm`: Estimated dimensions of the object in millimeters, formatted as a string "<W>x<H>x<D>" (e.g., "150x80x30").
    • W (width) = second-longest side of the object.
    • H (height) = longest side of the object.
    • D (depth) = shortest side of the object.
• `object_orientation`: The dominant orientation of the object as it appears in the image. Choose from: "horizontal", "vertical".

**Grasp Parameters (for the LEFT bionic hand):**
• `grasp_type`: Specify the type of grasp.
    • 0 = Palmar grasp (thumb opposes the palm side of the middle and/or ring fingers).
    • 1 = Lateral grasp (thumb opposes the radial side of the index finger, like holding a key or another thin object).
• `hand_rotation_deg`: Rotation of the hand in degrees, relative to the start pose (palm down, thumb pointing right), to appropriately grasp the object.
    • 0 = Palm remains down.
    • +90 = Palm rotates towards the left (handshake position, thumb points upwards).
    • +180 = Palm rotates to face upwards (thumb points towards the left).
    • -90 = Palm rotates towards the right (thumb points downwards).
    Interpolate for intermediate angles if necessary. Keep in mind that lateral grasp is done perpendicular to the object, to secure the object in between thumb and index finger.
• `hand_aperture_mm`: Estimate the required opening between the thumb and fingers in millimeters to encompass and securely hold the object. This should be a reasonable value for the object's size.
• `num_fingers`: The number of fingers (1, 2, 3, or 4) to use for the palmar grasp, where lateral grasp always uses 1, thumb only, excluding the thumb (the thumb is always used). Estimate this based on the height of the object's surface that the fingers will make contact with:
    • 1 finger: for object contact height up to 10mm covered by fingers.
    • 2 fingers: for object contact height >10mm and up to 25mm covered by fingers.
    • 3 fingers: for object contact height >25mm and up to 40mm covered by fingers.
    • 4 fingers: for object contact height >50mm covered by fingers.
    Always prioritize a stable grasp configuration appropriate for the object's shape and estimated size.

Example
-------
{"object_name":"mug","object_shape":"cylinder","object_dimensions_mm":"80x95x80","object_orientation":"vertical","grasp_type":1,"hand_rotation_deg":90,"hand_aperture_mm":75,"num_fingers":4}
"""


# --- 3. CHECKPOINT MANAGEMENT ---
def save_checkpoint(completed_tasks: Set[Tuple[str, str, str, int]], results: List[Dict[str, Any]]) -> None:
    """Save the current benchmark progress to a checkpoint file."""
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "completed_tasks": list(completed_tasks),
        "results": results
    }

    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Checkpoint saved to {CHECKPOINT_FILE}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def load_checkpoint() -> Tuple[Set[Tuple[str, str, str, int]], List[Dict[str, Any]]]:
    """Load benchmark progress from checkpoint file if it exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), []

    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)

        # Convert the list of completed tasks back to a set of tuples
        completed_tasks = set(tuple(task) for task in checkpoint_data.get("completed_tasks", []))
        results = checkpoint_data.get("results", [])

        logger.info(f"Loaded checkpoint from {CHECKPOINT_FILE} with {len(completed_tasks)} completed tasks")
        return completed_tasks, results
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return set(), []


# --- 4. RATE LIMIT HANDLING ---
def check_rate_limits() -> Dict[str, Any]:
    """Check the current rate limits and credits for the API key."""
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", {})
        else:
            logger.error(f"Failed to check rate limits: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Error checking rate limits: {e}")
        return {}


def adjust_workers_based_on_credits(rate_limit_info: Dict[str, Any]) -> int:
    """Adjust the number of worker threads based on available credits."""
    if not rate_limit_info:
        return MAX_WORKERS

    # Extract rate limit information
    usage = rate_limit_info.get("usage", 0)
    limit = rate_limit_info.get("limit")
    rate_limit = rate_limit_info.get("rate_limit", {})
    requests_per_interval = rate_limit.get("requests", 1)

    # If we have a credit limit, adjust workers based on remaining credits
    if limit is not None:
        remaining_credits = max(0, limit - usage)
        # Ensure we don't exceed our credits or the rate limit
        return min(remaining_credits, requests_per_interval, MAX_WORKERS)

    # If no limit (unlimited credits), use the rate limit or default
    return min(requests_per_interval, MAX_WORKERS)


# --- 5. HELPER FUNCTIONS ---
def encode_image_to_base64(image_path):
    """Encode image to base64 with caching to reduce redundant operations."""
    # Check if image is already in cache
    if image_path in image_cache and "base64" in image_cache[image_path]:
        return image_cache[image_path]["base64"]

    try:
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Initialize cache entry if not exists
            if image_path not in image_cache:
                image_cache[image_path] = {}

            # Store in cache
            image_cache[image_path]["base64"] = base64_data

            return base64_data
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None


def get_image_files(directory):
    image_files = []
    if not os.path.isdir(directory):
        logger.error(f"Error: Image directory '{directory}' not found or is not a directory.")
        return []
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(directory, filename))
    return image_files


def fetch_and_store_model_details():
    """Fetches pricing and other details for each model in MODELS_TO_TEST with better error handling."""
    global ALL_MODEL_DETAILS
    logger.info("Fetching model details from OpenRouter...")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    for model_id in MODELS_TO_TEST:
        url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                model_endpoint_data = response.json().get("data")

                if model_endpoint_data and model_endpoint_data.get("endpoints"):
                    # Use the pricing from the first listed endpoint/provider
                    first_endpoint = model_endpoint_data["endpoints"][0]
                    if "pricing" in first_endpoint:
                        pricing = first_endpoint["pricing"]

                        # Log the raw pricing data for debugging
                        logger.info(f"Raw pricing for {model_id}: {pricing}")

                        ALL_MODEL_DETAILS[model_id] = {
                            "id": model_id,
                            "name": model_endpoint_data.get("name", model_id),
                            "chosen_pricing": pricing,
                            "architecture": model_endpoint_data.get("architecture", {}),
                            # Store raw pricing for debugging
                            "raw_pricing": pricing
                        }
                    else:
                        logger.warning(f"No pricing info in the first endpoint for {model_id}.")
                        ALL_MODEL_DETAILS[model_id] = {"id": model_id, "name": model_id, "chosen_pricing": {}}
                else:
                    logger.warning(f"No endpoint data or endpoints list found for {model_id}.")
                    ALL_MODEL_DETAILS[model_id] = {"id": model_id, "name": model_id, "chosen_pricing": {}}
            else:
                logger.error(f"Error fetching details for {model_id}: {response.status_code} - {response.text}")
                ALL_MODEL_DETAILS[model_id] = {"id": model_id, "name": model_id, "chosen_pricing": {}}
        except Exception as e:
            logger.error(f"Error fetching details for {model_id}: {e}")
            ALL_MODEL_DETAILS[model_id] = {"id": model_id, "name": model_id, "chosen_pricing": {}}
        time.sleep(0.2)  # Be polite to the API

    if not ALL_MODEL_DETAILS:
        logger.warning("No model details fetched. Cost calculations will likely be zero.")
        return False
    logger.info(f"Fetched details for {len(ALL_MODEL_DETAILS)} models.")
    return True

def get_tokenizer_for_model(model_id):
    """Tries to get a tiktoken encoder based on model architecture (heuristic)."""
    if not TIKTOKEN_AVAILABLE:
        return None

    model_info = ALL_MODEL_DETAILS.get(model_id, {})
    architecture = model_info.get("architecture", {})
    tokenizer_name = architecture.get("tokenizer", "").lower()  # e.g., "GPT", "Gemini", "Claude"

    # Common tiktoken model names based on OpenAI's patterns
    # This is a very rough heuristic for OpenRouter models.
    if "gpt" in tokenizer_name or "openai" in model_id.lower() or "gpt-4o" in model_id.lower():
        try:
            return tiktoken.encoding_for_model("gpt-4o")  # A recent general one
        except:
            pass
    if "claude" in tokenizer_name or "anthropic" in model_id.lower():
        # Tiktoken doesn't directly support Claude. Anthropic has its own tokenizer.
        # For dry run, we might fall back to char count.
        return None
    return None


def estimate_input_tokens(prompt_text, model_id):
    # Try tiktoken for OpenAI models
    if TIKTOKEN_AVAILABLE and ("openai" in model_id.lower() or "gpt" in model_id.lower()):
        try:
            encoder = tiktoken.encoding_for_model("gpt-4o")
            tokens = len(encoder.encode(prompt_text))
            logger.debug(f"Tiktoken estimation for {model_id}: {tokens} tokens")
            return tokens
        except Exception as e:
            logger.debug(f"Tiktoken failed for {model_id}: {e}")

    # Model-specific token estimation based on known characteristics
    text_length = len(prompt_text)

    if "claude" in model_id.lower() or "anthropic" in model_id.lower():
        # Claude typically has ~3.5-4 chars per token
        estimated_tokens = text_length // 4 + 10
        logger.debug(f"Claude estimation for {model_id}: {estimated_tokens} tokens (chars: {text_length})")
        return estimated_tokens
    elif "gemini" in model_id.lower() or "google" in model_id.lower():
        # Gemini has different tokenization, roughly ~4-5 chars per token
        estimated_tokens = text_length // 4 + 10
        logger.debug(f"Gemini estimation for {model_id}: {estimated_tokens} tokens (chars: {text_length})")
        return estimated_tokens
    elif "mistral" in model_id.lower():
        # Mistral tokenization, roughly ~4 chars per token
        estimated_tokens = text_length // 4 + 10
        logger.debug(f"Mistral estimation for {model_id}: {estimated_tokens} tokens (chars: {text_length})")
        return estimated_tokens
    else:
        # Generic fallback
        estimated_tokens = text_length // 4 + 10
        logger.debug(f"Generic estimation for {model_id}: {estimated_tokens} tokens (chars: {text_length})")
        return estimated_tokens


def calculate_cost(model_id, prompt_tokens, completion_tokens, image_sent=False, is_dry_run=False):
    cost = 0.0
    model_details = ALL_MODEL_DETAILS.get(model_id)

    if not model_details or not model_details.get("chosen_pricing"):
        logger.debug(f"Pricing info not found for model {model_id}. Cost will be 0.")
        return 0.0

    pricing = model_details["chosen_pricing"]

    # Get raw pricing values
    price_prompt_raw = float(pricing.get("prompt", "0.0"))
    price_completion_raw = float(pricing.get("completion", "0.0"))
    price_per_image = float(pricing.get("image", "0.0"))
    price_per_request = float(pricing.get("request", "0.0"))

    # Log pricing details for debugging
    logger.debug(f"Model {model_id} pricing - Prompt: {price_prompt_raw}, "
                 f"Completion: {price_completion_raw}, Image: {price_per_image}, Request: {price_per_request}")

    if prompt_tokens is not None:
        prompt_cost = prompt_tokens * price_prompt_raw
        cost += prompt_cost
        logger.debug(f"Prompt cost: {prompt_tokens} tokens * {price_prompt_raw} = ${prompt_cost}")

    if not is_dry_run and completion_tokens is not None:
        completion_cost = completion_tokens * price_completion_raw
        cost += completion_cost
        logger.debug(f"Completion cost: {completion_tokens} tokens * {price_completion_raw} = ${completion_cost}")
    elif is_dry_run and completion_tokens is not None and completion_tokens > 0:
        completion_cost = completion_tokens * price_completion_raw
        cost += completion_cost
        logger.debug(
            f"Dry run completion cost: {completion_tokens} tokens * {price_completion_raw} = ${completion_cost}")

    # Image cost
    if image_sent:
        cost += price_per_image
        if price_per_image > 0:
            logger.debug(f"Adding image cost of ${price_per_image} for model {model_id}")

    # Request cost
    cost += price_per_request
    if price_per_request > 0:
        logger.debug(f"Adding request cost of ${price_per_request} for model {model_id}")

    logger.debug(f"Total cost for {model_id}: ${cost}")
    return cost


def validate_pricing_format():
    """
    Helper function to validate and log pricing format for all models.
    Call this after fetching model details to understand pricing structure.
    """
    logger.info("\n--- Pricing Format Validation ---")

    for model_id, details in ALL_MODEL_DETAILS.items():
        pricing = details.get("chosen_pricing", {})
        if not pricing:
            logger.warning(f"No pricing data for {model_id}")
            continue

        prompt_price = float(pricing.get("prompt", "0.0"))
        completion_price = float(pricing.get("completion", "0.0"))
        image_price = float(pricing.get("image", "0.0"))
        request_price = float(pricing.get("request", "0.0"))

        logger.info(f"{model_id}:")
        logger.info(f"  Prompt: ${prompt_price}")
        logger.info(f"  Completion: ${completion_price}")
        logger.info(f"  Image: ${image_price}")
        logger.info(f"  Request: ${request_price}")

        # Check if prices look like they're per million tokens
        if prompt_price > 0.001 or completion_price > 0.001:
            logger.warning(f"  ^ Prices for {model_id} might be per million tokens (values > 0.001)")
        elif 0 < prompt_price < 0.000001:
            logger.warning(f"  ^ Prices for {model_id} might be per individual token (very small values)")


def run_dry_run_cost_estimation():
    if not fetch_and_store_model_details():
        logger.error("Cannot perform dry run cost estimation without model pricing details.")
        return

    # Validate pricing format
    validate_pricing_format()

    image_paths = get_image_files(IMAGE_DIRECTORY)
    if not image_paths:
        logger.error("No images found for dry run estimation.")
        return

    total_estimated_cost = 0.0
    total_api_calls_to_be_made = 0

    logger.info("\n--- Improved Dry Run Cost Estimation ---")
    logger.info(
        f"Estimating for {len(image_paths)} images, {len(MODELS_TO_TEST)} models, {NUMBER_OF_RETRIES} retries per complex prompt.")
    logger.info(
        "Note: This estimates INPUT costs (text prompt, image, per-request). Output token costs are NOT included.")

    for model_id in MODELS_TO_TEST:
        model_name_for_print = ALL_MODEL_DETAILS.get(model_id, {}).get("name", model_id)
        logger.info(f"\n  Model: {model_name_for_print} ({model_id})")

        if model_id not in ALL_MODEL_DETAILS or not ALL_MODEL_DETAILS[model_id].get("chosen_pricing"):
            logger.warning(f"    Skipping {model_id} due to missing pricing info.")
            continue

        model_cost_estimate = 0.0

        # Cost for complex prompts (image + request + estimated input text tokens)
        for image_path in image_paths:
            for retry in range(NUMBER_OF_RETRIES):
                estimated_prompt_tokens = estimate_input_tokens(COMPLEX_PROMPT_TEXT, model_id)
                call_cost = calculate_cost(model_id, estimated_prompt_tokens, 0, image_sent=True,
                                                    is_dry_run=True)
                model_cost_estimate += call_cost
                total_api_calls_to_be_made += 1

        logger.info(f"    Estimated tokens per prompt: {estimate_input_tokens(COMPLEX_PROMPT_TEXT, model_id)}")
        logger.info(f"    Estimated cost for this model (all images & retries): ${model_cost_estimate:.6f}")
        total_estimated_cost += model_cost_estimate

    logger.info("\n--- Total Estimated Dry Run Cost (Input Costs Only) ---")
    logger.info(f"Estimated for {total_api_calls_to_be_made} potential API calls (if all run).")
    logger.info(f"Total Estimated Cost: ${total_estimated_cost:.6f}")
    logger.info("This does NOT include the cost of output/completion tokens from the models.")
    logger.info("Actual costs may vary based on precise tokenization by models and actual completion lengths.")

    # Provide cost per call breakdown
    if total_api_calls_to_be_made > 0:
        avg_cost_per_call = total_estimated_cost / total_api_calls_to_be_made
        logger.info(f"Average estimated cost per API call: ${avg_cost_per_call:.6f}")

def extract_json(text: str) -> str:
    """
    Remove ``` fences or leading labels and return the pure JSON string.
    """
    text = text.strip()
    if text.lower().startswith("response:"):
        text = text.split(":", 1)[1].lstrip()
    if text.startswith("```"):
        # Drop first line (``` or ```json ...)
        text = "\n".join(text.splitlines()[1:])
        # Drop trailing fence if present
        if text.rstrip().endswith("```"):
            text = "\n".join(text.splitlines()[:-1])
    return text.strip()


def process_complex_prompt(
        image_name: str,
        image_path: str,
        base64_image: str,
        model_id: str,
        retry_attempt: int,
        extra_headers: Dict[str, str]
) -> Dict[str, Any]:
    """Process a complex prompt using cached payloads when possible."""
    # Try to get cached payload template
    _, complex_payload_template = get_cached_payloads(image_path)

    if complex_payload_template:
        # Use cached payload directly
        messages_payload_complex = copy.deepcopy(complex_payload_template)
    else:
        # Fallback to direct payload creation if caching failed
        messages_payload_complex = [{"role": "user", "content": [
            {"type": "text", "text": COMPLEX_PROMPT_TEXT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}]

    start_time = time.time()
    error_message_complex = ""
    latency_complex = 0
    cost_complex = 0.0
    prompt_tokens_complex = 0
    completion_tokens_complex = 0
    total_tokens_complex = 0
    raw_response_complex = ""
    parsed_object_shape = "N/A"
    parsed_object_name = "N/A"
    parsed_object_dimensions = "N/A"
    parsed_object_orientation = "N/A"
    parsed_grasp_type = "N/A"
    parsed_hand_rotation = "N/A"
    parsed_hand_aperture = "N/A"
    parsed_num_fingers = "N/A"
    json_valid, keys_present = False, False

    # Create OpenAI client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    try:
        # Exponential backoff for retries on rate limit errors
        max_retries = RATE_LIMIT_MAX_RETRIES
        retry_count = 0
        retry_delay = 1  # Start with 1 second delay

        while retry_count < max_retries:
            try:
                completion_complex = client.chat.completions.create(
                    model=model_id,
                    messages=messages_payload_complex,
                    extra_headers=extra_headers
                )

                latency_complex = time.time() - start_time
                raw_response_complex = completion_complex.choices[
                    0].message.content if completion_complex.choices else ""

                if completion_complex.usage:
                    prompt_tokens_complex = completion_complex.usage.prompt_tokens
                    completion_tokens_complex = completion_complex.usage.completion_tokens
                    total_tokens_complex = completion_complex.usage.total_tokens

                cost_complex = calculate_cost(model_id, prompt_tokens_complex, completion_tokens_complex,
                                              image_sent=True)

                # Parse JSON response
                try:
                    cleaned_response = extract_json(raw_response_complex)
                    parsed_data = json.loads(cleaned_response)
                    json_valid = True
                    parsed_object_shape = parsed_data.get("object_shape")
                    parsed_object_name = parsed_data.get("object_name")
                    parsed_object_dimensions = parsed_data.get("object_dimensions_mm")
                    parsed_object_orientation = parsed_data.get("object_orientation")
                    parsed_grasp_type = parsed_data.get("grasp_type")
                    parsed_hand_rotation = parsed_data.get("hand_rotation_deg")
                    parsed_hand_aperture = parsed_data.get("hand_aperture_mm")
                    parsed_num_fingers = parsed_data.get("num_fingers")

                    required_keys = ["object_shape", "object_name", "object_dimensions_mm", "object_orientation",
                                     "grasp_type", "hand_rotation_deg", "hand_aperture_mm", "num_fingers"]
                    keys_present = all(key in parsed_data for key in required_keys)

                    if not isinstance(parsed_object_shape, str) and parsed_object_shape is not None:
                        error_message_complex += "Shape!str. "
                    if not isinstance(parsed_object_name, str) and parsed_object_name is not None:
                        error_message_complex += "Name!str. "
                    if not isinstance(parsed_object_dimensions, str) and parsed_object_dimensions is not None:
                        error_message_complex += "Dims!str. "
                    if not isinstance(parsed_object_orientation, str) and parsed_object_orientation is not None:
                        error_message_complex += "Orient!str. "
                    if not isinstance(parsed_grasp_type, int) and parsed_grasp_type is not None:
                        error_message_complex += "Grasp!int. "
                    if not isinstance(parsed_hand_rotation, (int, float)) and parsed_hand_rotation is not None:
                        error_message_complex += "HRot!num. "
                    if not isinstance(parsed_hand_aperture, (int, float)) and parsed_hand_aperture is not None:
                        error_message_complex += "HAper!num. "
                    if not isinstance(parsed_num_fingers, int) and parsed_num_fingers is not None:
                        error_message_complex += "Fingers!int. "
                except json.JSONDecodeError as je:
                    error_message_complex += f"JSONErr:{je}. "
                except Exception as pe:
                    error_message_complex += f"ParseErr:{pe}. "

                break  # Success, exit retry loop

            except Exception as e:
                error_text = str(e)

                # Check if it's a rate limit error
                if "429" in error_text or "rate limit" in error_text.lower():
                    if retry_count < max_retries:
                        logger.warning(
                            f"Rate limit hit for {model_id}, retrying in {retry_delay}s (attempt {retry_count + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= RATE_LIMIT_BACKOFF_BASE  # Exponential backoff
                        retry_count += 1
                        continue

                # If we get here, it's either not a rate limit error or we've exhausted retries
                latency_complex = time.time() - start_time
                error_message_complex += f"APIErr:{error_text}. "
                logger.error(
                    f"Error processing complex prompt for {image_name} with {model_id}: {error_message_complex}")
                break

    except Exception as e:
        latency_complex = time.time() - start_time
        error_message_complex += f"UnexpectedErr:{str(e)}. "
        logger.error(
            f"Unexpected error processing complex prompt for {image_name} with {model_id}: {error_message_complex}")

    return {
        "Image_Name": image_name,
        "Image_Path": image_path,
        "VLM_Model": model_id,
        "Retry_Attempt": retry_attempt,
        "Prompt_Type": "complex_json",
        "Prompt_Text": "COMPLEX_PROMPT_TEXT",
        "Raw_Response": raw_response_complex,
        "Parsed_Object_Name": parsed_object_name,
        "Parsed_Object_Shape": parsed_object_shape,
        "Parsed_Object_Dimensions_mm": parsed_object_dimensions,
        "Parsed_Object_Orientation": parsed_object_orientation,
        "Parsed_Grasp_Type": parsed_grasp_type,
        "Parsed_Hand_Rotation_Deg": parsed_hand_rotation,
        "Parsed_Hand_Aperture_mm": parsed_hand_aperture,
        "Parsed_Num_Fingers": parsed_num_fingers,
        "JSON_Valid": json_valid,
        "Required_Keys_Present": keys_present,
        "Error_Message": error_message_complex.strip(),
        "Latency_Seconds": round(latency_complex, 3),
        "Cost_USD": round(cost_complex, 8),
        "Prompt_Tokens": prompt_tokens_complex,
        "Completion_Tokens": completion_tokens_complex,
        "Total_Tokens": total_tokens_complex
    }


def run_benchmark(continue_from_checkpoint: bool = True, force_workers: int = None):
    """Run the benchmark with parallel requests and checkpoint recovery."""
    if not fetch_and_store_model_details():
        logger.error("Proceeding with actual run, but cost calculations might be zero due to model detail fetch error.")

    extra_headers = {}
    if HTTP_REFERER:
        extra_headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        extra_headers["X-Title"] = X_TITLE

    # Load checkpoint if continuing from previous run
    completed_tasks = set()
    results = []

    if continue_from_checkpoint:
        completed_tasks, results = load_checkpoint()
        if completed_tasks:
            logger.info(
                f"Continuing from checkpoint with {len(completed_tasks)} completed tasks and {len(results)} results")

    # Get image paths
    image_paths = get_image_files(IMAGE_DIRECTORY)
    if not image_paths:
        logger.error(f"No images found in directory for actual run: {IMAGE_DIRECTORY}")
        return

    logger.info(f"\n--- Starting Benchmark Run ---")
    logger.info(
        f"Processing {len(image_paths)} images with {len(MODELS_TO_TEST)} models, {NUMBER_OF_RETRIES} retries each.")

    # Check rate limits to determine optimal number of workers
    rate_limit_info = check_rate_limits()
    max_workers = force_workers if force_workers is not None else adjust_workers_based_on_credits(rate_limit_info)
    logger.info(f"Using {max_workers} worker threads for parallel requests")

    # Initialize an empty CSV file with headers if it doesn't exist
    if not os.path.exists(OUTPUT_CSV_FILE):
        write_results_to_csv([], CSV_HEADERS, OUTPUT_CSV_FILE)

    # Process all tasks in parallel across images and models
    all_tasks = []

    # Generate all tasks upfront, storing function+args pairs
    logger.info("Preparing and caching all images...")
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        base64_image = encode_image_to_base64(image_path)  # This will cache the base64 encoding

        if not base64_image:
            results.append({
                "Image_Name": image_name, "Image_Path": image_path, "VLM_Model": "N/A", "Retry_Attempt": 0,
                "Prompt_Type": "N/A", "Prompt_Text": "N/A", "Raw_Response": "N/A",
                "Parsed_Object_Name": "N/A", "Parsed_Object_Shape": "N/A", "Parsed_Object_Dimensions_mm": "N/A",
                "Parsed_Object_Orientation": "N/A", "Parsed_Grasp_Type": "N/A", "Parsed_Hand_Rotation_Deg": "N/A",
                "Parsed_Hand_Aperture_mm": "N/A", "Parsed_Num_Fingers": "N/A",
                "JSON_Valid": False, "Required_Keys_Present": False, "Error_Message": "Failed to encode image",
                "Latency_Seconds": 0, "Cost_USD": 0, "Prompt_Tokens": 0, "Completion_Tokens": 0, "Total_Tokens": 0
            })
            continue

        # Pre-cache the payloads
        get_cached_payloads(image_path)

        logger.info(f"Preparing tasks for image: {image_name}")

        # Generate tasks for each model and retry attempt
        for model_id in MODELS_TO_TEST:
            model_name_for_print = ALL_MODEL_DETAILS.get(model_id, {}).get("name", model_id)
            logger.info(f"  Adding tasks for model: {model_name_for_print}")

            if model_id not in ALL_MODEL_DETAILS or not ALL_MODEL_DETAILS[model_id].get("chosen_pricing"):
                logger.warning(
                    f"    Warning: Pricing info for '{model_id}' not properly fetched. Costs for this model may be 0.")

            # Generate tasks for each retry attempt
            for i in range(1, NUMBER_OF_RETRIES + 1):
                # Add task for complex prompt
                complex_task_key = (image_path, model_id, "complex_json", i)
                if complex_task_key not in completed_tasks:
                    all_tasks.append((process_complex_prompt, (
                        image_name, image_path, base64_image,
                        model_id, i, extra_headers
                    )))
                else:
                    logger.info(
                        f"      Skipping already completed task: complex_json for {image_name}, {model_id}, attempt {i}")

    # Run all tasks concurrently with rate limiting
    if all_tasks:
        logger.info(f"Executing {len(all_tasks)} tasks in parallel with {max_workers} workers")

        # Use concurrent.futures to run all tasks with controlled concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_task = {}
            for idx, (func, args) in enumerate(all_tasks):
                future = executor.submit(func, *args)
                future_to_task[future] = idx

            # Process results as they complete
            batch_size = 20  # Save results in batches to reduce file I/O
            batch_results = []
            completed_count = 0
            total_count = len(all_tasks)

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    batch_results.append(result)
                    completed_count += 1

                    # Create task key from result
                    task_key = (
                        result["Image_Path"],
                        result["VLM_Model"],
                        result["Prompt_Type"],
                        result["Retry_Attempt"]
                    )
                    completed_tasks.add(task_key)
                    results.append(result)

                    # Log progress periodically
                    if completed_count % 10 == 0 or completed_count == total_count:
                        logger.info(
                            f"Progress: {completed_count}/{total_count} tasks completed ({completed_count / total_count * 100:.1f}%)")

                    # Save results in batches to reduce disk I/O
                    if len(batch_results) >= batch_size or completed_count == total_count:
                        # Write batch to CSV
                        write_results_to_csv(batch_results, CSV_HEADERS, OUTPUT_CSV_FILE)

                        # Save checkpoint
                        save_checkpoint(completed_tasks, results)

                        # Clear batch
                        batch_results = []

                        # Check if we need to adjust workers based on rate limits
                        if not force_workers and completed_count % 50 == 0:
                            rate_limit_info = check_rate_limits()
                            new_max_workers = adjust_workers_based_on_credits(rate_limit_info)
                            if new_max_workers != max_workers:
                                logger.info(
                                    f"Adjusting worker count from {max_workers} to {new_max_workers} based on rate limits")
                                max_workers = new_max_workers
                                # Note: Can't dynamically adjust executor max_workers once it's created,
                                # but this will be used if we create a new executor

                except Exception as e:
                    logger.error(f"Error in task execution: {e}")

    logger.info("Benchmarking complete. Results saved.")

    # Remove checkpoint file if benchmark completed successfully
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info(f"Removed checkpoint file {CHECKPOINT_FILE} after successful completion")
    except Exception as e:
        logger.error(f"Error removing checkpoint file: {e}")


# Global variables for thread safety and caching
import threading

csv_write_lock = threading.Lock()

# Image cache to reduce memory usage and processing time
image_cache = {}


def get_cached_payloads(image_path: str):
    """Get cached message payloads for both simple and complex prompts."""
    if image_path not in image_cache or "payloads" not in image_cache[image_path]:
        # Ensure base64 encoding is cached
        b64 = encode_image_to_base64(image_path)
        if not b64:
            return None, None

        # Create payload templates
        simple = [
            {"role": "user", "content": [
                {"type": "text", "text": None},  # to be filled later with specific prompt
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]

        complex_ = [
            {"role": "user", "content": [
                {"type": "text", "text": COMPLEX_PROMPT_TEXT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]

        # Initialize cache entry if needed
        if image_path not in image_cache:
            image_cache[image_path] = {}

        # Cache the payloads
        image_cache[image_path]["payloads"] = {"simple": simple, "complex": complex_}

    return image_cache[image_path]["payloads"]["simple"], image_cache[image_path]["payloads"]["complex"]


def write_results_to_csv(results: List[Dict[str, Any]], csv_headers: List[str], output_file: str) -> None:
    """Write benchmark results to CSV file in a thread-safe manner."""
    if not results:  # Skip if no results to write
        return

    with csv_write_lock:
        try:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.isfile(output_file)

            # Use append mode to avoid overwriting existing results
            with open(output_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers, extrasaction='ignore')

                # Only write header if creating a new file
                if not file_exists:
                    writer.writeheader()

                for row in results:
                    writer.writerow(row)

            logger.info(f"Batch of {len(results)} results written to {output_file}")
        except Exception as e:
            logger.error(f"Error writing CSV file: {e}")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM Benchmarking or Dry Run Cost Estimation.")
    parser.add_argument('--mode', type=str, choices=['actual', 'dryrun'], required=True,
                        help="'actual' to run the full benchmark, 'dryrun' for cost estimation.")
    parser.add_argument('--restart', action='store_true',
                        help="Restart the benchmark from the beginning, ignoring any checkpoint.")
    parser.add_argument('--workers', type=int, default=None,
                        help="Force a specific number of worker threads instead of auto-adjusting based on rate limits.")

    args = parser.parse_args()

    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY" or not OPENROUTER_API_KEY:
        logger.error("!!! ERROR: Please set your OPENROUTER_API_KEY in the script. !!!")
    elif IMAGE_DIRECTORY == "path/to/your/images" or not os.path.isdir(IMAGE_DIRECTORY):
        logger.error(
            f"!!! ERROR: Please set a valid IMAGE_DIRECTORY. Current: '{IMAGE_DIRECTORY}' is not a valid directory. !!!")
    else:
        if args.mode == 'dryrun':
            logger.info("Selected mode: Dry Run Cost Estimation")
            run_dry_run_cost_estimation()
        elif args.mode == 'actual':
            logger.info("Selected mode: Actual Benchmark Run")
            continue_from_checkpoint = not args.restart
            if args.restart:
                logger.info("Restarting benchmark from the beginning (ignoring checkpoint)")
            run_benchmark(continue_from_checkpoint=continue_from_checkpoint, force_workers=args.workers)
