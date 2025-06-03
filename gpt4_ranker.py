import aiohttp
import openai
import asyncio
import os
import base64
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                return None

async def rank_images(images, history_folder, water_well_name=None, max_selections=10):
    tasks = [download_image(img["url"]) for img in images]
    raw_images = await asyncio.gather(*tasks)

    results = []
    for i, img_bytes in enumerate(raw_images):
        if not img_bytes:
            continue
        try:
            # Fix base64 encoding
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Enhanced prompt with water well context
            prompt = f"""Rank this image for importance to a donor for the {water_well_name or 'water well project'}. 
            Rate from 1-10 based on:
            - Clear plaques with donor names (high priority)
            - Smiling children using the well (high priority)  
            - Emotional content showing impact (medium priority)
            - Clear well construction/completion (medium priority)
            - Community gathering around well (low-medium priority)
            
            Provide a brief reason for the score."""
            
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
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
