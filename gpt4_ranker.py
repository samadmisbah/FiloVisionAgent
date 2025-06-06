import aiohttp
import asyncio
import os
import base64
import re
import json
from openai import AsyncOpenAI

# Add Google API imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    print("Google API client not available - history folder features disabled")
    GOOGLE_API_AVAILABLE = False

# Initialize OpenAI client with new syntax
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

async def get_history_examples(folder_id, max_examples=5):
    """
    Get example images from history folder for context
    """
    if not GOOGLE_API_AVAILABLE:
        print("Google API not available - skipping history examples")
        return ""
    
    try:
        # You'll need to implement Google Drive API access here
        # This is a placeholder - replace with your actual Google Drive logic
        print(f"🔍 Fetching history examples from folder: {folder_id}")
        
        # Placeholder implementation
        # Replace this with actual Google Drive API calls to:
        # 1. List files in the history folder
        # 2. Download a few example images
        # 3. Analyze them to provide context
        
        return "Previous successful photos showed children happily using the pump with clear water flow and readable donor plaques."
        
    except Exception as e:
        print(f"Error fetching history examples: {e}")
        return ""

async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    """
    Rank images using OpenAI Vision API for donor appeal
    """
    print(f"🎯 Starting image ranking for {water_well_name}")
    print(f"📊 Processing {len(images)} images")
    print(f"📂 History folder: {history_folder}")
    
    # Fix the NameError - use the images parameter
    valid_images = images
    
    if not valid_images:
        print("❌ No valid images provided")
        return {"error": "No valid images provided"}
    
    # Input validation section with EXPLICIT filename mapping
    input_validation = f"""
🔍 INPUT VALIDATION:
Total images to rank: {len(valid_images)}
"""
    
    # CREATE EXPLICIT FILENAME MAPPING - This is the key fix!
    filename_mapping = []
    for i, img in enumerate(valid_images):
        filename = img.get('name', img.get('filename', f'image_{i}.jpg'))
        img_id = img.get('id', f'unknown_id_{i}')
        input_validation += f"\n{i+1}. ID: {img_id}, Filename: {filename}"
        filename_mapping.append(f"Image {i+1}: id='{img_id}', filename='{filename}'")
    
    mapping_text = "\n".join(filename_mapping)
    
    # Process history folder if provided
    history_context = ""
    if history_folder and GOOGLE_API_AVAILABLE:
        try:
            print(f"🔍 Processing history folder: {history_folder}")
            # Extract folder ID from the URL
            folder_id_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', history_folder)
            if folder_id_match:
                folder_id = folder_id_match.group(1)
                print(f"📁 Extracted folder ID: {folder_id}")
                
                # Get history examples
                history_examples = await get_history_examples(folder_id)
                if history_examples:
                    history_context = f"""
📚 CONTEXT FROM PREVIOUS SUCCESSFUL WELL PHOTOS:
{history_examples}

Use these successful examples as reference for what donors find most appealing.
"""
                    print(f"✅ Added history context")
                else:
                    print("⚠️ No history examples found")
            else:
                print("❌ Could not extract folder ID from history URL")
        except Exception as e:
            print(f"❌ Error processing history folder: {e}")
            history_context = ""
    else:
        if not history_folder:
            print("📂 No history folder provided")
        if not GOOGLE_API_AVAILABLE:
            print("❌ Google API not available for history processing")
        history_context = ""

    # ENHANCED PROMPT with explicit filename mapping
    enhanced_prompt = f"""You are ranking {len(valid_images)} images from a water well project for donor appeal.

{input_validation}

{history_context}

🎯 GOAL: Identify the **top 2 donor images**:
- Priority {len(valid_images)} (becomes _1_): clearest plaque WITH children who are smiling, happy, or joyful — ideally around the plaque. Do NOT select a plaque-only image even if it's the clearest plaque. There must be joyful children around it for top priority.
- Priority {len(valid_images)-1} (becomes _2_): joyful interaction with water (children splashing, smiling, visibly enjoying)

You must assign a **priority score** to each image from 1 (lowest donor appeal) to {len(valid_images)} (highest donor appeal), using each number exactly once.

📊 DETAILED RANKING GUIDE (Donor Visual Preference):

🥇 {len(valid_images)}/10 — Plaque fully readable with donor name + joyful children in front holding water containers, excellent framing and lighting.  
🥈 {len(valid_images)-1}/10 — Happy children playing with or splashing water; very lively, natural joy, good clarity.  
🥉 {max(1, len(valid_images)-2)}/10 — Children operating the pump with clear water flow and happy expressions, medium-range framing.  
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

📎 CRITICAL FILENAME MAPPING - USE THESE EXACT VALUES:
{mapping_text}

❌ NEVER use generic names like "image1.jpg", "image2.jpg" 
✅ ONLY use the exact id and filename from the mapping above

Respond with ONLY a JSON array using the EXACT IDs and filenames from the mapping above:

[
  {{
    "id": "exact-id-from-mapping-above",
    "filename": "exact-filename-from-mapping-above",
    "priority": 1,
    "reason": "Short visual justification"
  }},
  ...
]

🔒 VALIDATION: Ensure every result uses an exact ID and filename from the mapping. Use each priority number 1-{len(valid_images)} once, with no duplicates or missing numbers."""

    # Download and encode images
    message_content = [{"type": "text", "text": enhanced_prompt}]
    
    print("🖼️ Processing images for Vision API...")
    for i, img in enumerate(valid_images):
        try:
            img_data = await download_image(img['url'])
            if img_data:
                base64_image = base64.b64encode(img_data).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
                print(f"✅ Added image {i+1}: {img.get('name', img.get('filename', 'Unknown'))}")
            else:
                print(f"❌ Failed to download image {i+1}: {img.get('name', img.get('filename', 'Unknown'))}")
        except Exception as e:
            print(f"❌ Error processing image {i+1}: {e}")

    # Call OpenAI Vision API with NEW SYNTAX
    try:
        print("🤖 Calling OpenAI Vision API with GPT-4o...")
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message_content}],
            max_tokens=2000,
            temperature=0.1
        )
        
        print("✅ Received response from OpenAI GPT-4o")
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return {"error": f"OpenAI API error: {str(e)}"}

    # Process and validate response
    try:
        # Extract JSON from the response
        response_text = response.choices[0].message.content
        print(f"📤 Raw OpenAI response: {response_text}")
        
        # Try to extract JSON array from the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            print(f"✅ Parsed {len(result)} ranked images")
            
            # Enhanced validation - check required fields
            for i, item in enumerate(result):
                if not all(key in item for key in ['id', 'filename', 'priority', 'reason']):
                    print(f"⚠️ Item {i} missing required fields: {item}")
                    return {"error": f"Missing required fields in result {i}", "item": item}
                    
            # Validate priorities are unique and complete
            priorities = [item.get('priority') for item in result]
            expected_priorities = list(range(1, len(valid_images) + 1))
            
            if sorted(priorities) != expected_priorities:
                print(f"⚠️ Priority validation failed!")
                print(f"   Expected: {expected_priorities}")
                print(f"   Got: {sorted(priorities)}")
                return {"error": "Priority numbers missing or duplicated", 
                       "expected": expected_priorities, "received": sorted(priorities)}
            else:
                print("✅ Priority validation passed")
            
            # CRITICAL: Validate filenames match input exactly
            input_filenames = [img.get('name', img.get('filename', '')) for img in valid_images]
            input_ids = [img.get('id', '') for img in valid_images]
            
            for item in result:
                returned_filename = item.get("filename", "")
                returned_id = item.get("id", "")
                
                if returned_filename not in input_filenames:
                    print(f"❌ Invalid filename returned: '{returned_filename}'")
                    print(f"   Valid filenames: {input_filenames}")
                    return {"error": f"Invalid filename returned: '{returned_filename}'. Must use one of: {input_filenames}"}
                    
                if returned_id not in input_ids:
                    print(f"❌ Invalid ID returned: '{returned_id}'")
                    print(f"   Valid IDs: {input_ids}")
                    return {"error": f"Invalid ID returned: '{returned_id}'. Must use one of: {input_ids}"}
            
            print("✅ Filename and ID validation passed")
            print(f"🎯 Returning {len(result)} ranked images to n8n")
            return result
            
        else:
            print("❌ Could not extract JSON from response")
            return {"error": "Could not parse JSON from response", "raw_response": response_text}
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return {"error": "Invalid JSON in response", "raw_response": response_text}
    except Exception as e:
        print(f"❌ Unexpected error processing response: {e}")
        return {"error": str(e), "raw_response": response_text}

def extract_score_from_caption(caption):
    """Extract numerical score from caption text"""
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
