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
            
            # Enhanced prompt with water well context and unique ranking requirement
            prompt = f"""COMBINED VISION AGENT RANKING INSTRUCTIONS
            WATER WELL PROJECT CONTEXT
            You are ranking images for importance to donors for the {water_well_name} water well project. Your primary goal is to select images that will maximize donor engagement and demonstrate project impact.
            DONOR PRIORITY SCORING CRITERIA:
            HIGH PRIORITY (8-10 points):

            Clear plaques with donor names prominently displayed
            Children smiling while using or near the well
            Families gathering around the completed well
            Clear water flowing from the well demonstrating success

            MEDIUM PRIORITY (5-7 points):

            Well construction progress showing community involvement
            Community leaders or elders present at the well
            Educational signage about water safety or well maintenance
            Before/after comparison showing transformation

            LOW-MEDIUM PRIORITY (3-4 points):

            General construction activities
            Equipment or materials being delivered
            Landscape/environmental context
            Technical aspects of well construction

            HISTORY FOLDER PROCESSING (CRITICAL):

            You will receive a history_folder field containing the path/reference to previous successful water well images
            STEP 1: READ the history folder images FIRST before ranking current images
            STEP 2: ANALYZE successful patterns in the history images:

            What types of shots were previously selected as top priority?
            What visual elements made those images donor-appealing?
            What composition, subjects, and contexts were favored?


            STEP 3: APPLY these learned patterns to rank current images
            STEP 4: DOCUMENT your findings in the response

            If history_folder is provided, you MUST:

            Load and examine ALL images in the history folder
            Identify the visual patterns of high-performing images
            Use these patterns as your PRIMARY ranking criteria
            Document which historical patterns influenced your rankings

            If history_folder is null/empty:

            Use the donor appeal criteria below as your primary guide
            Note in your response that no historical context was available

            CRITICAL SEQUENTIAL RANKING REQUIREMENTS:
            1. COUNT TOTAL IMAGES FIRST

            You will receive an array of images in the images field
            ALWAYS count the total number of images first: total_images = images.length
            Log this count: "Processing {total_images} images for {water_well_name}"

            2. MANDATORY UNIQUE SEQUENTIAL RANKING

            EVERY image MUST receive a UNIQUE priority number from 1 to N
            HIGHEST number = BEST donor appeal (e.g., priority 10 = best out of 10 images)
            LOWEST number = LEAST donor appeal (e.g., priority 1 = least appealing out of 10 images)
            Rankings MUST be sequential: 1, 2, 3, 4, 5... up to total_images
            NO duplicate priority numbers allowed - this is CRITICAL
            NO skipping numbers (e.g., don't go 1, 3, 5)

            3. RANKING SCALE EXAMPLES:

            5 images: Priority 5 = BEST donor appeal, Priority 1 = LEAST donor appeal
            10 images: Priority 10 = BEST donor appeal, Priority 1 = LEAST donor appeal
            15 images: Priority 15 = BEST donor appeal, Priority 1 = LEAST donor appeal

            4. RESPONSE FORMAT:
            json{
              "total_images_processed": 10,
              "water_well_name": "{water_well_name}",
              "history_folder_used": "folder_path_or_null",
              "history_analysis": {
                "images_analyzed": 15,
                "key_patterns_found": [
                  "Children interacting with well featured in 80% of top selections",
                  "Donor plaques visible in 60% of highest-ranked images",
                  "Community gathering shots consistently ranked high"
                ],
                "applied_to_current_ranking": true
              },
              "ranking_method": "history_informed_donor_appeal",
              "results": [
                {
                  "id": "file_id_here",
                  "filename": "original_filename.jpg",
                  "priority": 10,
                  "score": 9.8,
                  "reason": "Score: 10 - Children smiling while using well with donor plaque visible, matches successful pattern from history folder"
                },
                {
                  "id": "file_id_here", 
                  "filename": "another_filename.jpg",
                  "priority": 9,
                  "score": 8.5,
                  "reason": "Score: 9 - Clear water flowing, aligns with high-performing historical selections showing project success"
                }
                // ... continue for ALL images with unique priorities
              ]
            }
            5. VALIDATION REQUIREMENTS:

            ✅ MANDATORY: Read and analyze history folder images if provided
            ✅ Verify results.length === total_images_processed
            ✅ Verify all priority numbers from 1 to N are used exactly once
            ✅ Verify no duplicate priority values exist
            ✅ Verify every input image has a corresponding result
            ✅ Use water well donor appeal criteria for scoring
            ✅ Document how historical patterns influenced rankings
            ✅ Include history analysis in response

            6. CRITICAL SUCCESS FACTORS:

            Process history folder FIRST: Load and analyze historical successful images before ranking current ones
            Use unique rankings: Each image must get a different priority number
            Focus on donor appeal: Prioritize images that show impact and community benefit
            Learn from history: Apply patterns from past successful selections to current rankings
            Differentiate quality: Even similar images must get different rankings based on historical insights
            Complete coverage: Rank ALL input images, not just the best ones
            Document learning: Show how historical analysis influenced your decisions

            EXAMPLE FOR 7 IMAGES:
            Input: 7 images from water well project
            Required Output: 7 results with priorities 1, 2, 3, 4, 5, 6, 7

            Priority 7 (highest donor appeal) → will be renamed to "_1_filename"
            Priority 6 (second highest) → will be renamed to "_2_filename"
            Priority 1 (lowest donor appeal) → will be renamed to "_7_filename"

            FAILURE CONDITIONS TO AVOID:

            ❌ All images getting the same priority
            ❌ Missing priority numbers in sequence
            ❌ Using priority numbers outside 1-N range
            ❌ Returning fewer results than input images
            ❌ Ignoring water well donor appeal criteria

            REMEMBER: Your rankings directly impact which images donors see first. Make every ranking count for maximum project support!"""
            
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
