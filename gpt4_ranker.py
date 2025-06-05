import aiohttp
import openai
import asyncio
import os
import base64
import re
import json

# Add Google API imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    print("Google API client not available - history folder features disabled")
    GOOGLE_API_AVAILABLE = False

openai.api_key = os.getenv("OPENAI_API_KEY")

# Global cache for history examples
_history_cache = {}

async def download_image(url):
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            async with session.get(url, headers=headers) as resp:
                print(f"Download status: {resp.status} for {url}")
                
                if resp.status == 200:
                    content_type = resp.headers.get('content-type', '')
                    if 'image' in content_type.lower():
                        data = await resp.read()
                        print(f"Downloaded {len(data)} bytes")
                        return data
                    else:
                        print(f"Not an image content type: {content_type}")
                        return None
                else:
                    print(f"Failed to download: {resp.status}")
                    return None
    except Exception as e:
        print(f"Download error: {e}")
        return None

# ... (keep all existing functions like get_history_examples and image processing logic) ...

async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    # ... (same as before up to simple_prompt definition) ...

    simple_prompt = f"""You are ranking {len(valid_images)} images from a water well project for donor appeal.

{input_validation}

{history_context}

You must assign a **priority score** to each image from 1 (lowest donor appeal) to {len(valid_images)} (highest donor appeal), using each number exactly once.

📊 DETAILED RANKING GUIDE (Donor Visual Preference):

🥇 10/10 — Plaque fully readable with donor name + joyful children in front holding water containers, excellent framing and lighting.  
🥈 9/10 — Happy children playing with or splashing water; very lively, natural joy, good clarity.  
🥉 8/10 — Children operating the pump with clear water flow and happy expressions, medium-range framing.  
7/10 — Children actively filling pots from the pump; visible water and full-body shots, slightly less vibrant.  
6/10 — Kids drinking and filling simultaneously, joyful but cluttered or minor visibility issues.  
5/10 — Drinking from hands or group joy, but lighting or focus not optimal.  
4/10 — Mixed engagement; some expressions unclear or partially blocked subjects.  
3/10 — Pumping and drinking scene with dispersed subjects, plaque partially cut off.  
2/10 — Large group image with minimal interaction or emotional expression, plaque distant.  
1/10 — Children sitting or praying near the well; no water activity, static or subdued composition.

📌 TIP: Donor appeal is highest when:
- Plaque is clearly visible with donor name.
- Children look happy and actively use the well.
- Water flow is visible.
- The scene feels authentic, clean, and emotionally uplifting.

Respond with ONLY a JSON array using the EXACT IDs and filenames from the validation list above:
[
  {{"id": "{valid_images[0]['id']}", "filename": "{valid_images[0]['filename']}", "priority": 1, "reason": "Short reason"}},
  {{"id": "{valid_images[1]['id'] if len(valid_images) > 1 else 'EXAMPLE_ID'}", "filename": "{valid_images[1]['filename'] if len(valid_images) > 1 else 'EXAMPLE_FILENAME'}", "priority": 2, "reason": "Short reason"}}
]

🔒 VALIDATION: Ensure every result uses an exact ID and filename from the list. Use each priority number 1-{len(valid_images)} once, with no duplicates or missing numbers."""

    # ... (rest of the function continues unchanged including message_content, OpenAI call, response handling) ...

# Keep rest of your file unchanged

def extract_score_from_caption(caption):
    patterns = [
        r'(\d+)/10',
        r'[Ss]core:?\s*(\d+)',
        r'[Rr]ating:?\s*(\d+)',
        r'(\d+)\s*out of 10'
    ]

    for pattern in patterns:
        match = re.search(pattern, caption)
        if match:
            return int(match.group(1))
    return None
