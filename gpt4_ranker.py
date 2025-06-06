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

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ENHANCED: Global cache for history examples with one-time download
_history_cache = {}
_history_cache_lock = asyncio.Lock()

def get_google_drive_service():
    """Initialize Google Drive service using service account"""
    try:
        # You'll need to set up service account credentials
        service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        if not service_account_file:
            print("❌ GOOGLE_SERVICE_ACCOUNT_FILE environment variable not set")
            return None
            
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        service = build('drive', 'v3', credentials=credentials)
        print("✅ Google Drive service initialized")
        return service
        
    except Exception as e:
        print(f"❌ Error initializing Google Drive service: {e}")
        return None

async def get_all_images_from_folder(service, folder_id, max_images=20):
    """
    Recursively get all images from folder and ALL its subfolders
    """
    all_images = []
    folders_to_process = [folder_id]
    
    while folders_to_process and len(all_images) < max_images:
        current_folder = folders_to_process.pop(0)
        print(f"🔍 Scanning folder: {current_folder}")
        
        try:
            # Get all items in current folder
            results = service.files().list(
                q=f"'{current_folder}' in parents and trashed=false",
                fields="files(id, name, mimeType, webContentLink)",
                pageSize=100
            ).execute()
            
            items = results.get('files', [])
            print(f"📁 Found {len(items)} items in folder {current_folder}")
            
            for item in items:
                # If it's a folder, add to processing queue
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    folders_to_process.append(item['id'])
                    print(f"📂 Added subfolder to queue: {item['name']}")
                
                # If it's an image, add to results
                elif item['mimeType'].startswith('image/'):
                    # Create download URL
                    download_url = f"https://drive.google.com/uc?id={item['id']}&export=download"
                    all_images.append({
                        'id': item['id'],
                        'name': item['name'],
                        'url': download_url,
                        'folder_id': current_folder
                    })
                    print(f"🖼️ Found image: {item['name']}")
                    
                    if len(all_images) >= max_images:
                        break
                        
        except Exception as e:
            print(f"❌ Error scanning folder {current_folder}: {e}")
            continue
    
    print(f"✅ Total images found across all subfolders: {len(all_images)}")
    return all_images

async def download_and_analyze_history_images(history_images, max_analyze=5):
    """
    Download history images and analyze them for context
    """
    analyzed_examples = []
    
    # Limit to avoid token limits
    images_to_analyze = history_images[:max_analyze]
    
    for img in images_to_analyze:
        try:
            print(f"📥 Downloading history image: {img['name']}")
            img_data = await download_image(img['url'])
            
            if img_data:
                base64_image = base64.b64encode(img_data).decode('utf-8')
                
                # Quick analysis of this history image
                analysis_prompt = """Analyze this water well photo briefly. What makes it appealing for donors? 
                Focus on: children's engagement, water visibility, plaque readability, overall composition.
                Respond in 1-2 sentences."""
                
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"  # Low detail to save tokens
                                    }
                                }
                            ]
                        }],
                        max_tokens=150,
                        temperature=0.1
                    )
                    
                    analysis = response.choices[0].message.content
                    analyzed_examples.append({
                        'filename': img['name'],
                        'analysis': analysis
                    })
                    print(f"✅ Analyzed: {img['name']}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {img['name']}: {e}")
                    
        except Exception as e:
            print(f"❌ Error downloading {img['name']}: {e}")
    
    return analyzed_examples

async def get_history_examples(folder_id, max_examples=5):
    """
    Get example images from history folder for context - WITH GLOBAL CACHING
    """
    if not GOOGLE_API_AVAILABLE:
        print("Google API not available - skipping history examples")
        return ""
    
    # ONE-TIME DOWNLOAD: Check global cache first
    async with _history_cache_lock:
        cache_key = f"history_{folder_id}"
        
        if cache_key in _history_cache:
            print(f"✅ Using CACHED history examples for folder: {folder_id}")
            return _history_cache[cache_key]
        
        print(f"🔄 FIRST TIME: Downloading history examples from folder: {folder_id}")
        
        try:
            # Initialize Google Drive service
            service = get_google_drive_service()
            if not service:
                return ""
            
            # Get ALL images from ALL subfolders
            print(f"🔍 Scanning ALL subfolders in history folder: {folder_id}")
            history_images = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: asyncio.run(get_all_images_from_folder(service, folder_id, max_images=20))
            )
            
            if not history_images:
                print("❌ No history images found")
                _history_cache[cache_key] = ""
                return ""
            
            # Download and analyze selected images
            analyzed_examples = await download_and_analyze_history_images(
                history_images, 
                max_analyze=max_examples
            )
            
            if not analyzed_examples:
                print("❌ No history images could be analyzed")
                _history_cache[cache_key] = ""
                return ""
            
            # Create context string
            context_parts = ["📚 SUCCESSFUL EXAMPLES FROM PREVIOUS WELLS:"]
            for i, example in enumerate(analyzed_examples, 1):
                context_parts.append(f"{i}. {example['filename']}: {example['analysis']}")
            
            context_parts.append("\n🎯 Use these successful patterns to guide your ranking decisions.")
            
            history_context = "\n".join(context_parts)
            
            # CACHE THE RESULT for all subsequent calls
            _history_cache[cache_key] = history_context
            
            print(f"✅ CACHED history context ({len(analyzed_examples)} examples)")
            print(f"📊 Cache size: {len(history_context)} characters")
            
            return history_context
            
        except Exception as e:
            print(f"❌ Error processing history folder: {e}")
            _history_cache[cache_key] = ""  # Cache empty result to avoid retries
            return ""

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

async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    """
    Rank images using OpenAI Vision API for donor appeal
    """
    print(f"🎯 Starting image ranking for {water_well_name}")
    print(f"📊 Processing {len(images)} images")
    print(f"📂 History folder: {history_folder}")
    
    valid_images = images
    
    if not valid_images:
        print("❌ No valid images provided")
        return {"error": "No valid images provided"}
    
    # Input validation section with EXPLICIT filename mapping
    input_validation = f"""
🔍 INPUT VALIDATION:
Total images to rank: {len(valid_images)}
"""
    
    # CREATE EXPLICIT FILENAME MAPPING
    filename_mapping = []
    for i, img in enumerate(valid_images):
        filename = img.get('name', img.get('filename', f'image_{i}.jpg'))
        img_id = img.get('id', f'unknown_id_{i}')
        input_validation += f"\n{i+1}. ID: {img_id}, Filename: {filename}"
        filename_mapping.append(f"Image {i+1}: id='{img_id}', filename='{filename}'")
    
    mapping_text = "\n".join(filename_mapping)
    
    # Process history folder if provided - WITH CACHING
    history_context = ""
    if history_folder and GOOGLE_API_AVAILABLE:
        try:
            print(f"🔍 Processing history folder: {history_folder}")
            # Extract folder ID from the URL
            folder_id_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', history_folder)
            if folder_id_match:
                folder_id = folder_id_match.group(1)
                print(f"📁 Extracted folder ID: {folder_id}")
                
                # Get history examples (cached after first call)
                history_examples = await get_history_examples(folder_id)
                if history_examples:
                    history_context = f"\n{history_examples}\n"
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

    # ENHANCED PROMPT with history context
    enhanced_prompt = f"""You are ranking {len(valid_images)} images from a water well project for donor appeal.

{input_validation}

{history_context}

🚨 CRITICAL RULE: NEVER rank a water well by itself (no children visible) as Priority {len(valid_images)}. Images with ONLY the water well structure and NO children must be ranked 3 or lower.

🎯 GOAL: Identify the **top 2 donor images**:
- Priority {len(valid_images)} (becomes _1_): MUST have both readable plaque AND joyful children visible in the same frame. If no image has both plaque + children, choose the image with happiest children interacting with water.
- Priority {len(valid_images)-1} (becomes _2_): joyful interaction with water (children splashing, smiling, visibly enjoying)

You must assign a **priority score** to each image from 1 (lowest donor appeal) to {len(valid_images)} (highest donor appeal), using each number exactly once.

📊 DETAILED RANKING GUIDE (Donor Visual Preference):

🚨 FORBIDDEN: Water well alone (no children) = MAX Priority 3 
🥇 Priority {len(valid_images)} — Plaque readable + joyful children together in same frame, excellent lighting
🥈 Priority {len(valid_images)-1} — Happy children playing with/splashing water, very lively and natural joy
🥉 Priority {max(1, len(valid_images)-2)} — Children operating pump with clear water flow and happy expressions
Priority 7 — Children filling containers from pump, visible water, full-body shots
Priority 6 — Kids drinking and filling simultaneously, joyful but cluttered  
Priority 5 — Drinking from hands or group joy, suboptimal lighting/focus
Priority 4 — Mixed engagement, some unclear expressions or blocked subjects
Priority 3 — MAXIMUM for well-only images or pumping with dispersed/unhappy children
Priority 2 — Large group with minimal interaction, plaque distant
Priority 1 — Static composition, no water activity, subdued or unclear

⛔ ABSOLUTE RULE: Images showing ONLY the water well structure without any children visible CANNOT be ranked higher than Priority 3, regardless of plaque clarity.

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

    # Download and encode current images for ranking
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

    # Call OpenAI Vision API
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

    # Process and validate response (rest of the function remains the same)
    try:
        response_text = response.choices[0].message.content
        print(f"📤 Raw OpenAI response: {response_text}")
        
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
