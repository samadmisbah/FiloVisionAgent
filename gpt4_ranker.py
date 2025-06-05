import aiohttp
import openai
import asyncio
import os
import base64
import re
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

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
    NEW: Batch processing approach - rank ALL images together
    """
    
    # Extract enhanced payload parameters
    total_image_count = kwargs.get('total_image_count', len(images))
    ranking_requirements = kwargs.get('ranking_requirements', {})
    instructions = kwargs.get('instructions', '')
    
    print(f"=== VISION AGENT PROCESSING ===")
    print(f"Total images to rank: {total_image_count}")
    print(f"Water well: {water_well_name}")
    print(f"History folder: {history_folder}")
    print(f"Enhanced instructions provided: {bool(instructions)}")
    
    # Download all images
    print("Downloading all images...")
    tasks = [download_image(img["url"]) for img in images]
    raw_images = await asyncio.gather(*tasks)
    
    # Prepare images for batch analysis
    valid_images = []
    failed_images = []
    
    for i, img_bytes in enumerate(raw_images):
        if not img_bytes or len(img_bytes) < 100:
            failed_images.append({
                "index": i,
                "filename": images[i]["name"], 
                "id": images[i].get("id", "unknown"),
                "error": "Failed to download or invalid image"
            })
            continue
            
        try:
            # Validate image format
            if not (img_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                   img_bytes.startswith(b'\x89PNG') or        # PNG
                   img_bytes.startswith(b'GIF87a') or         # GIF
                   img_bytes.startswith(b'GIF89a') or         # GIF
                   img_bytes.startswith(b'RIFF')):            # WebP
                raise Exception("Invalid image format")
            
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
            
            valid_images.append({
                "index": i,
                "filename": images[i]["name"],
                "id": images[i].get("id", "unknown"),
                "b64": img_b64,
                "mime_type": mime_type
            })
            
        except Exception as e:
            failed_images.append({
                "index": i,
                "filename": images[i]["name"],
                "id": images[i].get("id", "unknown"), 
                "error": str(e)
            })
    
    print(f"Valid images: {len(valid_images)}, Failed: {len(failed_images)}")
    
    if not valid_images:
        print("No valid images to process!")
        return []
    
    # NEW: Batch ranking prompt
    batch_prompt = f"""You are ranking {len(valid_images)} images for a water well project: "{water_well_name}".

CRITICAL REQUIREMENTS:
1. You MUST rank ALL {len(valid_images)} images with UNIQUE priority numbers from 1 to {len(valid_images)}
2. Priority {len(valid_images)} = BEST image, Priority 1 = WORST image  
3. NO duplicate priorities allowed
4. You must return a JSON response with exactly {len(valid_images)} results

RANKING CRITERIA (in order of importance):
HIGH PRIORITY (best images):
- Clear plaques with donor names prominently displayed
- Children smiling while using or near the well
- Families gathering around the completed well
- Clear water flowing from the well demonstrating success

MEDIUM PRIORITY:
- Well construction progress showing community involvement
- Community leaders or elders present at the well
- Educational signage about water safety or well maintenance
- Before/after comparison showing transformation

LOW PRIORITY:
- General construction activities
- Equipment or materials being delivered
- Landscape/environmental context
- Technical aspects of well construction

RESPONSE FORMAT - Return valid JSON:
{{
  "total_images_processed": {len(valid_images)},
  "water_well_name": "{water_well_name}",
  "results": [
    {{"id": "image_id", "filename": "image_name", "priority": 1, "score": 1.0, "reason": "Description"}},
    {{"id": "image_id", "filename": "image_name", "priority": 2, "score": 2.0, "reason": "Description"}},
    ...continue for all {len(valid_images)} images with unique priorities 1-{len(valid_images)}
  ]
}}

Analyze all images and rank them from 1 (worst) to {len(valid_images)} (best) for donor appeal."""
    
    # Prepare message with all images
    message_content = [{"type": "text", "text": batch_prompt}]
    
    # Add all images to the message
    for img in valid_images:
        message_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:{img['mime_type']};base64,{img['b64']}"}
        })
    
    try:
        print("Sending batch request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": message_content
            }],
            max_tokens=2000  # Increased for batch response
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"OpenAI Response: {response_text[:500]}...")  # First 500 chars
        
        # Parse JSON response
        try:
            ranking_data = json.loads(response_text)
            results = ranking_data.get("results", [])
            
            # Validate response
            if len(results) != len(valid_images):
                raise Exception(f"Expected {len(valid_images)} results, got {len(results)}")
            
            # Check for duplicate priorities
            priorities = [r.get("priority") for r in results]
            if len(set(priorities)) != len(priorities):
                raise Exception("Duplicate priorities detected!")
            
            # Map results back to original indices and add failed images
            final_results = []
            
            # Add successful rankings
            for result in results:
                final_results.append({
                    "filename": result["filename"],
                    "priority": result["priority"],
                    "reason": result["reason"],
                    "id": result["id"],
                    "score": result.get("score", result["priority"])
                })
            
            # Add failed images with priority 0
            for failed in failed_images:
                final_results.append({
                    "filename": failed["filename"],
                    "priority": 0,  # Lowest priority for failed images
                    "reason": f"Failed to process: {failed['error']}",
                    "id": failed["id"],
                    "score": 0
                })
            
            print(f"Successfully ranked {len(final_results)} images")
            return final_results
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response_text}")
            raise Exception("Invalid JSON response from OpenAI")
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback: return simple sequential ranking
        fallback_results = []
        for i, img in enumerate(valid_images):
            fallback_results.append({
                "filename": img["filename"],
                "priority": i + 1,  # Sequential priority
                "reason": f"Fallback ranking #{i + 1} due to processing error: {str(e)}",
                "id": img["id"],
                "score": i + 1
            })
        
        # Add failed images
        for failed in failed_images:
            fallback_results.append({
                "filename": failed["filename"],
                "priority": 0,
                "reason": f"Failed to process: {failed['error']}",
                "id": failed["id"],
                "score": 0
            })
        
        return fallback_results

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
