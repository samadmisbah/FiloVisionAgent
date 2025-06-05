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
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            async with session.get(url, headers=headers) as resp:
                print(f"Download status: {resp.status} for {url}")
                print(f"Content-Type: {resp.headers.get('content-type', 'unknown')}")
                
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
    Simplified batch processing with better JSON handling
    """
    
    # Extract enhanced payload parameters
    total_image_count = kwargs.get('total_image_count', len(images))
    
    print(f"=== VISION AGENT PROCESSING ===")
    print(f"Total images to rank: {total_image_count}")
    print(f"Water well: {water_well_name}")
    print(f"History folder: {history_folder}")
    
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
    
    # SIMPLIFIED: Limit to 6 images max for batch processing to avoid token limits
    if len(valid_images) > 6:
        print(f"Too many images ({len(valid_images)}), processing first 6 only")
        valid_images = valid_images[:6]
    
    # Simplified batch prompt for better JSON response
    batch_prompt = f"""Rank these {len(valid_images)} water well images from 1 (worst) to {len(valid_images)} (best) for donor appeal.

CRITERIA:
- High priority: Donor plaques, children using well, families, water flowing
- Medium priority: Construction with community, leaders present
- Low priority: Equipment, landscape, technical aspects

Return ONLY this JSON format:
[
  {{"id": "image_id_1", "filename": "name1", "priority": 1, "reason": "Brief reason"}},
  {{"id": "image_id_2", "filename": "name2", "priority": 2, "reason": "Brief reason"}},
  {{"id": "image_id_3", "filename": "name3", "priority": 3, "reason": "Brief reason"}}
]

Use priorities 1 to {len(valid_images)} exactly once each."""
    
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
            max_tokens=1500,
            temperature=0.1  # Lower temperature for more consistent JSON
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"OpenAI Response length: {len(response_text)} chars")
        print(f"Response preview: {response_text[:200]}...")
        
        # Try to extract JSON from response
        try:
            # Look for JSON array in the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                print(f"Extracted JSON: {json_str[:200]}...")
                
                ranking_data = json.loads(json_str)
                
                if isinstance(ranking_data, list) and len(ranking_data) == len(valid_images):
                    print(f"✅ Successfully parsed {len(ranking_data)} rankings")
                    
                    # Validate unique priorities
                    priorities = [r.get("priority") for r in ranking_data]
                    if len(set(priorities)) == len(priorities):
                        print("✅ All priorities are unique")
                        
                        # Convert to final format
                        final_results = []
                        for result in ranking_data:
                            final_results.append({
                                "filename": result["filename"],
                                "priority": result["priority"],
                                "reason": result["reason"],
                                "id": result["id"],
                                "score": result["priority"]
                            })
                        
                        # Add failed images with priority 0
                        for failed in failed_images:
                            final_results.append({
                                "filename": failed["filename"],
                                "priority": 0,
                                "reason": f"Failed: {failed['error']}",
                                "id": failed["id"],
                                "score": 0
                            })
                        
                        print(f"✅ Returning {len(final_results)} total results")
                        return final_results
                    else:
                        raise Exception("Duplicate priorities found")
                else:
                    raise Exception(f"Expected {len(valid_images)} results, got {len(ranking_data) if isinstance(ranking_data, list) else 'non-list'}")
            else:
                raise Exception("No JSON array found in response")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            raise Exception(f"Invalid JSON: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        print(f"Full response was: {response_text if 'response_text' in locals() else 'No response'}")
        
        # IMPROVED FALLBACK: Use simple heuristic ranking
        print("Using improved fallback ranking...")
        fallback_results = []
        
        for i, img in enumerate(valid_images):
            # Simple heuristic: images with certain keywords get higher priority
            filename_lower = img["filename"].lower()
            
            # High priority keywords
            if any(word in filename_lower for word in ['children', 'family', 'plaque', 'donor', 'celebration']):
                base_priority = len(valid_images) - (i // 3)  # Top third
            # Medium priority keywords  
            elif any(word in filename_lower for word in ['community', 'group', 'ceremony', 'opening']):
                base_priority = len(valid_images) // 2 + (i % 3)  # Middle
            else:
                base_priority = 1 + (i % 3)  # Lower priority
            
            # Ensure unique priorities
            final_priority = max(1, min(len(valid_images), base_priority))
            
            fallback_results.append({
                "filename": img["filename"],
                "priority": final_priority,
                "reason": f"Heuristic ranking based on filename analysis. Error: {str(e)[:100]}",
                "id": img["id"],
                "score": final_priority
            })
        
        # Ensure unique priorities by adjusting duplicates
        used_priorities = set()
        for result in fallback_results:
            original_priority = result["priority"]
            while result["priority"] in used_priorities:
                result["priority"] = result["priority"] + 1 if result["priority"] < len(valid_images) else 1
            used_priorities.add(result["priority"])
        
        # Add failed images
        for failed in failed_images:
            fallback_results.append({
                "filename": failed["filename"],
                "priority": 0,
                "reason": f"Failed: {failed['error']}",
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
