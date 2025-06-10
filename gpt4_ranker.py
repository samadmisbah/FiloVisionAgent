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
                Focus on: children's engagement with clean water, water visibility and flow, plaque readability, overall composition.
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

# NEW: Content validation function for rule enforcement
async def validate_image_content(image_data, filename):
    """
    Quick content validation for rule enforcement
    """
    try:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        validation_prompt = """Look at this water well image and answer YES or NO:

1. CHILDREN_VISIBLE: Are children clearly visible?
2. PLAQUE_READABLE: Is a donation plaque/sign readable?
3. WATER_ACTIVE: Is clean water flowing or being actively used?
4. CHILDREN_HAPPY: Do children appear happy/smiling?
5. WELL_ONLY: Is this ONLY the well structure with NO children?

Format: CHILDREN_VISIBLE: YES/NO, PLAQUE_READABLE: YES/NO, WATER_ACTIVE: YES/NO, CHILDREN_HAPPY: YES/NO, WELL_ONLY: YES/NO"""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": validation_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }],
            max_tokens=50,
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content
        print(f"🔍 Content validation for {filename}: {response_text}")
        
        return {
            'has_children': 'CHILDREN_VISIBLE: YES' in response_text,
            'has_plaque': 'PLAQUE_READABLE: YES' in response_text,
            'has_water': 'WATER_ACTIVE: YES' in response_text,
            'children_happy': 'CHILDREN_HAPPY: YES' in response_text,
            'well_only': 'WELL_ONLY: YES' in response_text
        }
        
    except Exception as e:
        print(f"❌ Content validation error for {filename}: {e}")
        return {'has_children': False, 'has_plaque': False, 'has_water': False, 'children_happy': False, 'well_only': False}

# NEW: Step 3 - Automatic priority fixing function
def fix_duplicate_priorities(result, total_images):
    """
    STEP 3: Automatically fix duplicate priorities by reassigning them sequentially
    """
    print("🔧 FIXING DUPLICATE PRIORITIES...")
    
    # Sort by current priority (highest first), then by filename for consistency
    sorted_result = sorted(result, key=lambda x: (-x['priority'], x['filename']))
    
    # Reassign priorities 1 to N sequentially
    for i, item in enumerate(sorted_result):
        new_priority = total_images - i  # Highest priority = total_images
        old_priority = item['priority']
        item['priority'] = new_priority
        
        if old_priority != new_priority:
            print(f"   🔄 {item['filename']}: {old_priority} → {new_priority}")
            # Update reason to reflect the fix
            item['reason'] = f"FIXED: {item['reason']}"
    
    print("✅ Priority fixing completed")
    return sorted_result

# NEW: Rule enforcement function
async def enforce_ranking_rules(result, image_validations, total_images):
    """
    Enforce strict ranking rules on the AI result
    """
    print("🚨 ENFORCING STRICT RANKING RULES...")
    
    # Find violations and ideal candidates
    violations = []
    ideal_top = []
    ideal_second = []
    well_only = []
    
    for item in result:
        img_id = item['id']
        priority = item['priority']
        validation = image_validations.get(img_id, {})
        
        # Check for well-only violation in top positions
        if validation.get('well_only', False) and priority > 3:
            violations.append(f"❌ {item['filename']}: Well-only image ranked {priority} (max allowed: 3)")
            well_only.append(item)
        
        # Find ideal candidates
        elif validation.get('has_plaque', False) and validation.get('children_happy', False):
            ideal_top.append(item)
        elif validation.get('children_happy', False) and validation.get('has_water', False):
            ideal_second.append(item)
    
    # If violations found, fix them
    if violations:
        print("⚠️ RULE VIOLATIONS DETECTED:")
        for violation in violations:
            print(f"   {violation}")
        
        # Get current top 2
        result_sorted = sorted(result, key=lambda x: x['priority'], reverse=True)
        top_item = result_sorted[0] if len(result_sorted) > 0 else None
        second_item = result_sorted[1] if len(result_sorted) > 1 else None
        
        # Fix top position if needed
        if top_item and image_validations.get(top_item['id'], {}).get('well_only', False):
            # Find best replacement
            if ideal_top:
                replacement = ideal_top[0]
                print(f"🔄 FIXING: Moving {replacement['filename']} to top position")
                # Swap priorities
                old_priority = replacement['priority']
                replacement['priority'] = top_item['priority']
                top_item['priority'] = min(old_priority, 3)
                replacement['reason'] = "🥇 ENFORCED: Top donor image with plaque + children"
                top_item['reason'] = f"🚫 ENFORCED: Well-only limited to priority {top_item['priority']}"
            elif ideal_second:
                replacement = ideal_second[0]
                print(f"🔄 FIXING: Moving {replacement['filename']} to top position")
                old_priority = replacement['priority']
                replacement['priority'] = top_item['priority']
                top_item['priority'] = min(old_priority, 3)
                replacement['reason'] = "🥇 ENFORCED: Top donor image with happy children + water"
                top_item['reason'] = f"🚫 ENFORCED: Well-only limited to priority {top_item['priority']}"
        
        # Fix second position if needed
        if second_item and image_validations.get(second_item['id'], {}).get('well_only', False):
            remaining_candidates = [img for img in (ideal_top + ideal_second) if img['priority'] != total_images]
            if remaining_candidates:
                replacement = remaining_candidates[0]
                print(f"🔄 FIXING: Moving {replacement['filename']} to second position")
                old_priority = replacement['priority']
                replacement['priority'] = second_item['priority']
                second_item['priority'] = min(old_priority, 3)
                replacement['reason'] = "🥈 ENFORCED: Second donor image with children interaction"
                second_item['reason'] = f"🚫 ENFORCED: Well-only limited to priority {second_item['priority']}"
        
        print("✅ RULE ENFORCEMENT COMPLETED")
    else:
        print("✅ No rule violations detected")
    
    return result

async def handle_content_policy_violation(valid_images, enhanced_prompt, water_well_name):
    """
    Handle content policy violations by processing images individually
    """
    print("🔧 HANDLING CONTENT POLICY VIOLATION...")
    print("🔍 Testing images individually to identify problematic ones...")
    
    safe_images = []
    problematic_images = []
    
    # Test each image individually
    for i, img in enumerate(valid_images):
        try:
            print(f"🧪 Testing image {i+1}: {img.get('name', 'Unknown')}")
            
            # Download and encode just this image
            img_data = await download_image(img['url'])
            if not img_data:
                print(f"❌ Could not download image {i+1}")
                problematic_images.append(img)
                continue
                
            base64_image = base64.b64encode(img_data).decode('utf-8')
            
            # Simple test prompt
            test_prompt = f"Analyze this clean water well image briefly for donor appeal. Rate 1-10 and explain why children are interacting with the clean water source."
            
            test_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": test_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }],
                max_tokens=100,
                temperature=0.1
            )
            
            test_result = test_response.choices[0].message.content
            
            if "I'm sorry, I can't assist" in test_result or "I cannot" in test_result:
                print(f"🚨 PROBLEMATIC IMAGE IDENTIFIED: {img.get('name', 'Unknown')}")
                problematic_images.append(img)
            else:
                print(f"✅ Safe image: {img.get('name', 'Unknown')}")
                safe_images.append(img)
                
        except Exception as e:
            print(f"❌ Error testing image {i+1}: {e}")
            problematic_images.append(img)
            
        # Add small delay to avoid rate limits
        await asyncio.sleep(0.5)
    
    print(f"📊 CONTENT POLICY ANALYSIS COMPLETE:")
    print(f"   ✅ Safe images: {len(safe_images)}")
    print(f"   🚨 Problematic images: {len(problematic_images)}")
    
    if len(safe_images) == 0:
        return {
            "error": "All images failed content policy check",
            "details": f"All {len(valid_images)} images were flagged by OpenAI content policy",
            "problematic_files": [img.get('name', 'Unknown') for img in problematic_images]
        }
    
    if len(problematic_images) > 0:
        print(f"🔄 PROCEEDING WITH {len(safe_images)} SAFE IMAGES...")
        print("🚨 Problematic images excluded:")
        for img in problematic_images:
            print(f"   - {img.get('name', 'Unknown')}")
    
    # Proceed with safe images only
    return await rank_safe_images(safe_images, enhanced_prompt, water_well_name, problematic_images)

async def rank_safe_images(safe_images, enhanced_prompt, water_well_name, excluded_images):
    """
    Rank only the safe images after content policy filtering
    """
    print(f"🎯 RANKING {len(safe_images)} SAFE IMAGES...")
    
    # Update prompt for reduced image count
    total_safe = len(safe_images)
    updated_prompt = enhanced_prompt.replace(
        f"You are ranking {len(safe_images) + len(excluded_images)} images",
        f"You are ranking {total_safe} images (some excluded by content policy)"
    )
    
    # Update priority ranges in prompt using regex
    updated_prompt = re.sub(
        r'Priority \d+',
        f'Priority {total_safe}',
        updated_prompt
    )
    updated_prompt = re.sub(
        r'priorities 1-\d+',
        f'priorities 1-{total_safe}',
        updated_prompt
    )
    updated_prompt = re.sub(
        r'Priority {len\(valid_images\)}',
        f'Priority {total_safe}',
        updated_prompt
    )
    
    # Create mapping for safe images only
    safe_mapping = []
    for i, img in enumerate(safe_images):
        filename = img.get('name', img.get('filename', f'image_{i}.jpg'))
        img_id = img.get('id', f'unknown_id_{i}')
        safe_mapping.append(f"Image {i+1}: id='{img_id}', filename='{filename}'")
    
    # Replace the mapping section
    mapping_section = "\n".join(safe_mapping)
    updated_prompt = re.sub(
        r'📎 CRITICAL FILENAME MAPPING.*?(?=\n\n|$)',
        f"📎 CRITICAL FILENAME MAPPING - USE THESE EXACT VALUES:\n{mapping_section}",
        updated_prompt,
        flags=re.DOTALL
    )
    
    # Download and encode safe images
    message_content = [{"type": "text", "text": updated_prompt}]
    image_data_map = {}
    
    for i, img in enumerate(safe_images):
        try:
            img_data = await download_image(img['url'])
            if img_data:
                image_data_map[img['id']] = img_data
                base64_image = base64.b64encode(img_data).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
                print(f"✅ Added safe image {i+1}: {img.get('name', 'Unknown')}")
        except Exception as e:
            print(f"❌ Error processing safe image {i+1}: {e}")
    
    # Call OpenAI with safe images
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message_content}],
            max_tokens=2000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        print(f"✅ Successfully ranked {len(safe_images)} safe images")
        
        # Process the response normally
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            # Validate and fix priorities for safe images
            priorities = [item.get('priority') for item in result]
            expected_priorities = list(range(1, len(safe_images) + 1))
            
            if sorted(priorities) != expected_priorities:
                print(f"🚨 DUPLICATE PRIORITIES IN SAFE IMAGES - FIXING...")
                result = fix_duplicate_priorities(result, len(safe_images))
            
            # Content validation and rule enforcement for safe images
            print("🔍 VALIDATING SAFE IMAGE CONTENT...")
            image_validations = {}
            
            for img in safe_images:
                if img['id'] in image_data_map:
                    validation = await validate_image_content(
                        image_data_map[img['id']], 
                        img.get('name', img.get('filename', 'Unknown'))
                    )
                    image_validations[img['id']] = validation
            
            # ENFORCE RANKING RULES on safe images
            result = await enforce_ranking_rules(result, image_validations, len(safe_images))
            
            # Add excluded images info to result
            if excluded_images:
                for item in result:
                    item['content_policy_note'] = f"{len(excluded_images)} images excluded by content policy"
                
                # Add summary of excluded files
                result.append({
                    "excluded_images": [img.get('name', 'Unknown') for img in excluded_images],
                    "exclusion_reason": "OpenAI content policy violation",
                    "total_processed": len(safe_images),
                    "total_excluded": len(excluded_images)
                })
            
            return result
        else:
            return {"error": "Could not parse JSON from safe images response", "raw_response": response_text}
            
    except Exception as e:
        return {"error": f"Error ranking safe images: {str(e)}"}

async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    """
    Rank images using OpenAI Vision API for donor appeal with BULLETPROOF validation and content policy handling
    """
    # Handle nested array structure from n8n
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], dict):
        if 'images' in images[0]:
            # Extract from nested structure: [{"images": [...], "history_folder": "..."}]
            data = images[0]
            images = data.get('images', [])
            history_folder = history_folder or data.get('history_folder')
            water_well_name = water_well_name or data.get('water_well_name')
            max_selections = max_selections or data.get('max_selections', 10)
            print("✅ Extracted parameters from nested structure")
    
    print(f"🎯 Starting BULLETPROOF image ranking for {water_well_name}")
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

    # STEP 1: ENHANCED PROMPT with balanced emphasis
    enhanced_prompt = f"""🚨 CRITICAL WATER WELL DONOR RANKING - STRICT RULES MUST BE FOLLOWED 🚨

You are ranking {len(valid_images)} images from a water well project for MAXIMUM donor appeal. This is for charity fundraising.

IMPORTANT: Assign each image a unique priority number from 1 to {len(valid_images)}. Use each number exactly once (no duplicates). Ignore any numbers in filenames - rank based on visual content only.

{input_validation}

{history_context}

🚨 ABSOLUTE REQUIREMENTS - NEVER VIOLATE THESE:

1. 🥇 Priority {len(valid_images)} (becomes _1_): 
   - MUST show readable donation plaque AND happy children together
   - If no image has both, choose HAPPIEST children actively using water
   - NEVER assign this to well-only images

2. 🥈 Priority {len(valid_images)-1} (becomes _2_):  
   - Happy children splashing/drinking clean water/playing with water
   - Children must show visible joy and engagement with clean water
   - Active clean water interaction required

3. 🚫 FORBIDDEN: Images with ONLY water well structure (no children) = MAXIMUM Priority 3

📊 DETAILED RANKING GUIDE (Donor Visual Preference):

🚨 FORBIDDEN: Water well alone (no children) = MAX Priority 3 
🥇 Priority {len(valid_images)} — Plaque readable + joyful children together in same frame, excellent lighting, perfect donor appeal
🥈 Priority {len(valid_images)-1} — Happy children playing with/splashing clean water, very lively and natural joy, clear engagement with water source
🥉 Priority {max(1, len(valid_images)-2)} — Children operating pump with clear clean water flow and happy expressions, good composition
Priority 7+ — Children filling containers from pump with clean water, visible water flow, full-body shots, positive energy
Priority 6 — Kids drinking clean water and filling containers simultaneously, joyful but may be cluttered  
Priority 5 — Drinking clean water from hands or group joy around water source, suboptimal lighting/focus but positive
Priority 4 — Mixed engagement, some unclear expressions or blocked subjects, minimal water interaction
Priority 3 — MAXIMUM for well-only images OR pumping with dispersed/unhappy children
Priority 2 — Large group with minimal clean water interaction, plaque distant, static feeling
Priority 1 — Very static composition, no clean water activity, subdued or unclear expressions

⛔ ABSOLUTE RULE: Images showing ONLY the water well structure without any children visible CANNOT be ranked higher than Priority 3, regardless of plaque clarity or well beauty.

📌 DONOR APPEAL MAXIMIZERS:
- Readable donor plaque with clear name visibility
- Children showing genuine happiness and joy around clean water
- Active clean water usage (flowing, splashing, drinking clean water from well)
- Clean, bright, emotionally uplifting scenes showing water access success
- Natural, candid moments vs posed shots

📎 CRITICAL FILENAME MAPPING - USE THESE EXACT VALUES:
{mapping_text}

❌ NEVER use generic names like "image1.jpg", "image2.jpg" 
✅ ONLY use the exact id and filename from the mapping above

Respond with ONLY a JSON array using the EXACT IDs and filenames from the mapping above. Assign priorities 1-{len(valid_images)} based on donor appeal, not filename numbers:

[
  {{
    "id": "exact-id-from-mapping-above",
    "filename": "exact-filename-from-mapping-above",
    "priority": UNIQUE_NUMBER_FROM_1_TO_{len(valid_images)},
    "reason": "Specific visual justification following the criteria above"
  }},
  ...
]

🔒 FINAL VALIDATION: Ensure every result uses an exact ID and filename from the mapping. Use each priority number 1-{len(valid_images)} once, with no duplicates or missing numbers."""

    # Download and encode current images for ranking - ENHANCED WITH DATA STORAGE
    message_content = [{"type": "text", "text": enhanced_prompt}]
    image_data_map = {}  # Store image data for validation
    
    print("🖼️ Processing images for Vision API...")
    for i, img in enumerate(valid_images):
        try:
            img_data = await download_image(img['url'])
            if img_data:
                # Store for later validation
                image_data_map[img['id']] = img_data
                
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

    # Call OpenAI Vision API with retry logic and content policy handling
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            print(f"🤖 Calling OpenAI Vision API with GPT-4o (attempt {attempt + 1})...")
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": message_content}],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Check for content policy violation
            response_text = response.choices[0].message.content
            if "I'm sorry, I can't assist with that" in response_text or "I cannot" in response_text:
                print("🚨 CONTENT POLICY VIOLATION DETECTED")
                print(f"   Response: {response_text}")
                
                # Try processing images individually to identify problematic ones
                return await handle_content_policy_violation(valid_images, enhanced_prompt, water_well_name)
            
            print("✅ Received response from OpenAI GPT-4o")
            break
            
        except Exception as e:
            print(f"❌ OpenAI API error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                return {"error": f"OpenAI API error after {max_retries + 1} attempts: {str(e)}"}
            await asyncio.sleep(2)  # Wait before retry

    # Process and validate response with BULLETPROOF validation
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
            
            # STEP 2: DUPLICATE DETECTION with automatic fixing
            priorities = [item.get('priority') for item in result]
            expected_priorities = list(range(1, len(valid_images) + 1))
            
            if sorted(priorities) != expected_priorities:
                print(f"🚨 STEP 2: DUPLICATE PRIORITIES DETECTED!")
                print(f"   Expected: {expected_priorities}")
                print(f"   Got: {sorted(priorities)}")
                
                # STEP 3: AUTOMATIC FIXING
                print("🔧 STEP 3: AUTOMATICALLY FIXING PRIORITIES...")
                result = fix_duplicate_priorities(result, len(valid_images))
                
                # Verify fix worked
                fixed_priorities = [item.get('priority') for item in result]
                if sorted(fixed_priorities) == expected_priorities:
                    print("✅ Priority fixing successful!")
                else:
                    print("❌ Priority fixing failed!")
                    return {"error": "Could not fix duplicate priorities", 
                           "expected": expected_priorities, "received": sorted(fixed_priorities)}
            else:
                print("✅ Priority validation passed - no duplicates detected")
            
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

            # Content validation and rule enforcement
            print("🔍 VALIDATING IMAGE CONTENT...")
            image_validations = {}
            
            for img in valid_images:
                if img['id'] in image_data_map:
                    validation = await validate_image_content(
                        image_data_map[img['id']], 
                        img.get('name', img.get('filename', 'Unknown'))
                    )
                    image_validations[img['id']] = validation
            
            # ENFORCE RANKING RULES
            result = await enforce_ranking_rules(result, image_validations, len(valid_images))
            
            # Log final ranking for verification
            print("📊 FINAL BULLETPROOF RANKING:")
            for item in sorted(result, key=lambda x: x['priority'], reverse=True):
                print(f"   Priority {item['priority']}: {item['filename']} - {item['reason']}")
            
            print(f"🎯 Returning {len(result)} BULLETPROOF RANKED images to n8n")
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
