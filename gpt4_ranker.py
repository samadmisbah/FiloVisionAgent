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

async def get_history_examples(history_folder):
    """
    Get ranked example images from Google Drive history folder structure
    History folder contains subfolders (different wells), each with ranked images
    Uses caching to avoid re-downloading same examples multiple times
    """
    if not history_folder or not GOOGLE_API_AVAILABLE:
        if not history_folder:
            print("No history folder provided")
        else:
            print("Google API not available - skipping history examples")
        return []
    
    # Check cache first
    if history_folder in _history_cache:
        print(f"Using cached history examples ({len(_history_cache[history_folder])} examples)")
        return _history_cache[history_folder]
    
    try:
        print(f"Fetching history examples from: {history_folder}")
        
        # Extract folder ID if it's a full URL
        folder_id = history_folder
        if 'folders/' in history_folder:
            folder_id = history_folder.split('folders/')[-1]
        elif 'id=' in history_folder:
            folder_id = history_folder.split('id=')[-1]
        
        print(f"History parent folder ID: {folder_id}")
        
        # Set up Google Drive API service
        credentials_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        if not credentials_json:
            print("No Google service account credentials found")
            return []
        
        # Parse credentials
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        # Build the Drive service
        drive_service = build('drive', 'v3', credentials=credentials)
        
        print("Listing subfolders in history folder...")
        
        # First, get all subfolders in the history folder
        subfolders_results = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
            fields='files(id,name)',
            pageSize=50
        ).execute()
        
        subfolders = subfolders_results.get('files', [])
        print(f"Found {len(subfolders)} subfolders in history folder")
        
        # Collect all ranked files from all subfolders
        all_ranked_files = {}
        
        for subfolder in subfolders:
            subfolder_id = subfolder['id']
            subfolder_name = subfolder['name']
            print(f"Checking subfolder: {subfolder_name}")
            
            # List files in this subfolder
            files_results = drive_service.files().list(
                q=f"'{subfolder_id}' in parents and trashed=false",
                fields='files(id,name)',
                pageSize=100
            ).execute()
            
            files = files_results.get('files', [])
            
            # Find ranked files in this subfolder
            for file in files:
                filename = file['name'].lower()
                # Look for pattern like _1_, _2_, _10_, etc.
                import re
                match = re.match(r'^_(\d+)_', filename)
                if match:
                    rank_num = int(match.group(1))
                    # Check if it's an image
                    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff']
                    extension = filename.split('.')[-1] if '.' in filename else ''
                    if extension in image_extensions:
                        # Store with rank as key (will naturally override duplicates with latest found)
                        all_ranked_files[rank_num] = {
                            'id': file['id'],
                            'name': file['name'],
                            'rank': rank_num,
                            'subfolder': subfolder_name,
                            'url': f"https://drive.google.com/uc?id={file['id']}&export=view"
                        }
                        print(f"Found ranked example: _{rank_num}_ - {file['name']} (from {subfolder_name})")
        
        # Sort by rank and take up to 10 examples
        sorted_ranks = sorted(all_ranked_files.keys())[:10]
        target_files = [all_ranked_files[rank] for rank in sorted_ranks]
        
        if not target_files:
            print("No ranked example files found in any history subfolders")
            return []
        
        print(f"Downloading {len(target_files)} history examples (ranks {sorted_ranks})...")
        
        # Download the example images
        history_examples = []
        for file_info in target_files:
            try:
                img_bytes = await download_image(file_info['url'])
                if img_bytes and len(img_bytes) > 100:
                    # Convert to base64
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Determine MIME type
                    if img_bytes.startswith(b'\xff\xd8\xff'):
                        mime_type = "image/jpeg"
                    elif img_bytes.startswith(b'\x89PNG'):
                        mime_type = "image/png"
                    elif img_bytes.startswith(b'GIF'):
                        mime_type = "image/gif"
                    elif img_bytes.startswith(b'RIFF'):
                        mime_type = "image/webp"
                    else:
                        mime_type = "image/jpeg"
                    
                    history_examples.append({
                        'name': file_info['name'],
                        'rank': file_info['rank'],
                        'subfolder': file_info['subfolder'],
                        'b64': img_b64,
                        'mime_type': mime_type
                    })
                    print(f"✅ Downloaded history example: {file_info['name']} (rank {file_info['rank']})")
                else:
                    print(f"❌ Failed to download: {file_info['name']}")
            except Exception as e:
                print(f"❌ Error downloading {file_info['name']}: {e}")
        
        print(f"Successfully loaded {len(history_examples)} history examples from {len(set(ex['subfolder'] for ex in history_examples))} different wells")
        
        # Cache the results for future requests
        _history_cache[history_folder] = history_examples
        print(f"Cached {len(history_examples)} history examples for future use")
        
        return history_examples
        
    except Exception as e:
        print(f"Error accessing history folder: {e}")
        return []

async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    """
    Simple, focused ranking with history folder examples
    """
    
    total_image_count = kwargs.get('total_image_count', len(images))
    
    print(f"=== SIMPLE VISION RANKING ===")
    print(f"Water well: {water_well_name}")
    print(f"Total images: {total_image_count}")
    print(f"History folder: {history_folder}")
    
    # Download all images (no artificial limits)
    print("Downloading all images...")
    tasks = [download_image(img["url"]) for img in images]
    raw_images = await asyncio.gather(*tasks)
    
    # Process images
    valid_images = []
    failed_images = []
    
    for i, img_bytes in enumerate(raw_images):
        if not img_bytes or len(img_bytes) < 100:
            failed_images.append({
                "filename": images[i]["name"], 
                "id": images[i].get("id", "unknown"),
                "priority": 0,
                "reason": "Failed to download",
                "score": 0
            })
            continue
            
        try:
            # Validate and convert image
            if not (img_bytes.startswith(b'\xff\xd8\xff') or img_bytes.startswith(b'\x89PNG') or 
                   img_bytes.startswith(b'GIF') or img_bytes.startswith(b'RIFF')):
                raise Exception("Invalid image format")
            
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            if img_bytes.startswith(b'\xff\xd8\xff'):
                mime_type = "image/jpeg"
            elif img_bytes.startswith(b'\x89PNG'):
                mime_type = "image/png"
            elif img_bytes.startswith(b'GIF'):
                mime_type = "image/gif"
            elif img_bytes.startswith(b'RIFF'):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
            
            valid_images.append({
                "filename": images[i]["name"],  # Keep original filename exactly
                "original_filename": images[i]["name"],  # Store backup copy
                "id": images[i].get("id", "unknown"),
                "b64": img_b64,
                "mime_type": mime_type
            })
            
        except Exception as e:
            failed_images.append({
                "filename": images[i]["name"],
                "id": images[i].get("id", "unknown"),
                "priority": 0,
                "reason": f"Processing error: {str(e)}",
                "score": 0
            })
    
    print(f"Valid images: {len(valid_images)}, Failed: {len(failed_images)}")
    
    if not valid_images:
        return failed_images
    
    # Get history context and examples
    history_examples = await get_history_examples(history_folder)
    history_context = ""
    
    if history_folder:
        if history_examples:
            num_examples = len(history_examples)
            wells_represented = len(set(ex['subfolder'] for ex in history_examples))
            history_context = f"\n\nIMPORTANT REFERENCE EXAMPLES:\nThe first {num_examples} images below are RANKED examples from {wells_represented} different water wells in your history folder. These show the COMPLETE QUALITY SPECTRUM from best (rank 1) to lower ranks that donors prefer. Study the full ranking pattern and apply the same quality standards to rank the NEW images (shown after the examples).\n\n"
        else:
            history_context = f"\n\nIMPORTANT: This ranking should match the style and quality of images in the reference folder '{history_folder}'. Look for similar composition, subjects, and donor appeal factors that have been successful before."
    
    # MUCH SIMPLER PROMPT with history context
    simple_prompt = f"""You are ranking {len(valid_images)} images for a water well project.

{history_context}

Rank from 1 (least donor appeal) to {len(valid_images)} (most donor appeal).

PRIORITY SCORING GUIDE:
🥇 PERFECT 10/10 (Highest Priority): Donor plaque is FULLY VISIBLE with ALL text clearly legible + happy children actively using the well + excellent composition and lighting.

🥈 EXCELLENT 9/10 (Second Priority): Happy children actively using/celebrating around the well + genuine joyful interaction + good quality, but plaque may be partially visible or not the focal point.

Other good elements:
- Clear water flowing
- Community gathering
- Families around well
- Well functionality shown

{f"After the reference examples, you will see the NEW images to rank. Apply the same quality standards you observe in the examples." if history_examples else ""}

Respond with ONLY a JSON array like this:
[
  {{"id": "{valid_images[0]['id']}", "filename": "{valid_images[0]['filename']}", "priority": 1, "reason": "Short reason"}},
  {{"id": "{valid_images[1]['id'] if len(valid_images) > 1 else 'example'}", "filename": "{valid_images[1]['filename'] if len(valid_images) > 1 else 'example.jpg'}", "priority": 2, "reason": "Short reason"}}
]

CRITICAL: Use the EXACT original filename provided. Do not modify, shorten, or change the filename in any way.

Use each priority number 1-{len(valid_images)} exactly once."""

    # Process all images at once
    message_content = [{"type": "text", "text": simple_prompt}]
    
    # Add history examples first if available
    if history_examples:
        print(f"Adding {len(history_examples)} history examples to prompt")
        for example in history_examples:
            message_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:{example['mime_type']};base64,{example['b64']}"}
            })
    
    # Then add the images to be ranked
    for img in valid_images:
        message_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:{img['mime_type']};base64,{img['b64']}"}
        })
    
    try:
        print(f"Sending all {len(valid_images)} images to OpenAI...")
        if history_examples:
            print(f"Including {len(history_examples)} history examples as reference")
            
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message_content}],
            max_tokens=2000,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"Response preview: {response_text[:100]}...")
        
        # Extract JSON
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            all_results = json.loads(json_str)
            
            if len(all_results) == len(valid_images):
                # Add score field and validate filenames
                for i, result in enumerate(all_results):
                    result['score'] = result['priority']
                    
                    # Ensure we have the original filename
                    if 'filename' not in result or not result['filename']:
                        result['filename'] = valid_images[i]['original_filename']
                        print(f"⚠️ Fixed missing filename for result {i}")
                    
                print(f"✅ Successfully processed all {len(all_results)} images")
            else:
                raise Exception(f"Expected {len(valid_images)} results, got {len(all_results)}")
        else:
            raise Exception("No JSON found in response")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        # Simple fallback - ensure original filenames are preserved
        all_results = []
        for i, img in enumerate(valid_images):
            all_results.append({
                "filename": img["original_filename"],  # Use exact original filename
                "priority": i + 1,
                "reason": f"Fallback ranking due to error: {str(e)[:50]}",
                "id": img["id"],
                "score": i + 1
            })
    
    # Add failed images
    all_results.extend(failed_images)
    
    print(f"✅ Total results: {len(all_results)}")
    return all_results

def extract_score_from_caption(caption):
    """Extract numeric score from GPT response"""
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
