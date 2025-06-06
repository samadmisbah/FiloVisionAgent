import aiohttp
import asyncio
import os
import base64
import re
import json
from openai import AsyncOpenAI

# Optional Google API setup (disabled unless configured)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def download_image(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get('content-type', '')
                    if 'image' in content_type.lower():
                        return await resp.read()
        return None
    except Exception as e:
        print(f"Download error: {e}")
        return None

async def get_history_examples(folder_id):
    if not GOOGLE_API_AVAILABLE:
        return ""
    try:
        return "Past successful images show donor plaque clearly and joyful children interacting with water."
    except Exception as e:
        print(f"History example error: {e}")
        return ""

async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    if not images:
        return {"error": "No valid images provided"}

    history_context = ""
    if history_folder and GOOGLE_API_AVAILABLE:
        folder_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', history_folder)
        if folder_match:
            folder_id = folder_match.group(1)
            history_context = await get_history_examples(folder_id)

    prompt = f"""
You are ranking {len(images)} water well images for donor appeal. Assign a unique priority score from 1 (worst) to {len(images)} (best), using each number once.

🎯 GOAL: Identify the **top 2 donor images**:
- `_1_`: clearest plaque with children around it
- `_2_`: joyful interaction with water (children splashing, smiling, visibly enjoying)

🧠 RANKING RULES:
✅ Rank highest:
- Children directly playing with water or holding containers
- Plaque is readable and framed well
- Natural joy and expressive faces
- Clean background, clear lighting

❌ Rank lower (3–{len(images)}):
- No water flow or joy
- Faces are turned, bored, or unclear
- Plaque cut off or out of frame
- Crowded, blurry, redundant, or awkward composition

🧪 Priority Definitions:
{len(images)} → Best image for `_1_`
{len(images)-1} → Best image for `_2_`
1 → Worst image in batch (static, joyless, poor visibility)

Return only a strict JSON array:
[
  {{
    "id": "image-id",
    "filename": "original.jpg",
    "priority": 1–{len(images)},
    "reason": "Short visual justification"
  }},
  ...
]
"""

    message_content = [{"type": "text", "text": prompt}]
    for img in images:
        try:
            img_data = await download_image(img['url'])
            if img_data:
                b64 = base64.b64encode(img_data).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high"
                    }
                })
        except Exception as e:
            print(f"Image error: {e}")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message_content}],
            max_tokens=2000,
            temperature=0.1
        )
    except Exception as e:
        return {"error": f"OpenAI error: {str(e)}"}

    try:
        text = response.choices[0].message.content
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return {"error": "No valid JSON array found in response", "raw": text}

        result = json.loads(match.group(0))

        # Validate unique priorities
        priorities = [item.get("priority") for item in result]
        expected = list(range(1, len(images) + 1))
        if sorted(priorities) != expected:
            return {"error": "Priority numbers missing or duplicated", "priorities": priorities}

        return result
    except Exception as e:
        return {"error": f"Result parsing failed: {e}"}
