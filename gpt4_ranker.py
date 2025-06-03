import aiohttp
import openai
import asyncio
import os
import base64
import re

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

async def rank_images(images, history_folder, water_well_name=None, max_selections=10):
    tasks = [download_image(img["url"]) for img in images]
    raw_images = await asyncio.gather(*tasks)

    results = []
    for i, img_bytes in enumerate(raw_images):
        if not img_bytes:
            results.append({
                "filename": images[i]["name"],
                "priority": 1,
                "reason": "Failed to download image",
                "id": images[i].get("id", "unknown"),
                "score": 1
            })
            continue
            
        try:
            # Validate we have actual image data
            if len(img_bytes) < 100:  # Too small to be a real image
                raise Exception("Downloaded file too small, likely not an image")
            
            # Check for common image file signatures
            if not (img_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                   img_bytes.startswith(b'\x89PNG') or        # PNG
                   img_bytes.startswith(b'GIF87a') or         # GIF
                   img_bytes.startswith(b'GIF89a') or         # GIF
                   img_bytes.startswith(b'RIFF')):            # WebP
                raise Exception("Downloaded data doesn't appear to be a valid image")
            
            # Convert to base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Determine image format for proper MIME type
            if img_bytes.startswith(b'\xff\xd8\xff'):
                mime_type = "image/jpeg"
            elif img_bytes.startswith(b'\x89PNG'):
                mime_type = "image/png"
            elif img_bytes.startswith(b'GIF'):
                mime_type = "image/gif"
            elif img_bytes.startswith(b'RIFF'):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default fallback
            
            # Enhanced prompt with water well context
            prompt = f"""Rank this image for importance to a donor for the {water_well_name or 'water well project'}. 
            Rate from 1-10 based on these priorities:

            HIGH PRIORITY (8-10 points):
            - Clear plaques with donor names prominently displayed
            - Children smiling while using or near the well
            - Families gathering around the completed well
            - Clear water flowing from the well demonstrating success

            MEDIUM PRIORITY (5-7 points):
            - Well construction progress showing community involvement
            - Community leaders or elders present at the well
            - Educational signage about water safety or well maintenance
            - Before/after comparison showing transformation

            LOW-MEDIUM PRIORITY (3-4 points):
            - General construction activities
            - Equipment or materials being delivered
            - Landscape/environmental context
            - Technical aspects of well construction

            Provide a score (1-10) and brief reason focusing on donor appeal and community impact."""
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=150
            )
            
            caption = response.choices[0].message.content.strip()
            
            # Extract score if provided, otherwise use position
            score = extract_score_from_caption(caption) or (10 - i)  # Higher score = higher priority
            
            results.append({
                "filename": images[i]["name"],
                "priority": score,
                "reason": caption,
                "id": images[i].get("id", "unknown"),
                "score": score
            })
            
        except Exception as e:
            results.append({
                "filename": images[i]["name"],
                "priority": 1,  # Low priority for errors
                "reason": f"Error analyzing image: {str(e)}",
                "id": images[i].get("id", "unknown"),
                "score": 1
            })
    
    # Sort by score (highest first) and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_selections]

def extract_score_from_caption(caption):
    """Extract numeric score from GPT response"""
    # Look for patterns like "8/10", "Score: 7", "Rating: 9"
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
