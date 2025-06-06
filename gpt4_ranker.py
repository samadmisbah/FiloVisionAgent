async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    if not images:
        return {"error": "No valid images provided"}

    history_context = ""
    if history_folder and GOOGLE_API_AVAILABLE:
        folder_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', history_folder)
        if folder_match:
            folder_id = folder_match.group(1)
            history_context = await get_history_examples(folder_id)

    # Create a mapping of input images for validation - CRITICAL for filename preservation
    input_images_map = {}
    for i, img in enumerate(images):
        original_name = img.get('name', img.get('filename', f'unknown_image_{i}'))
        input_images_map[str(i+1)] = {
            'original_name': original_name,
            'id': img.get('id', f'unknown_id_{i}'),
            'url': img.get('url', '')
        }
    
    prompt = f"""
🚨 CRITICAL FILENAME PRESERVATION RULES:
You are receiving {len(images)} images. For each image, you MUST use the EXACT filename provided.

INPUT IMAGE MAPPING (DO NOT DEVIATE):
{chr(10).join([f"Image {idx}: filename='{data['original_name']}', id='{data['id']}'" for idx, data in input_images_map.items()])}

❌ FORBIDDEN: Never use generic names like "image1.jpg", "image2.jpg", "photo1.jpg"
✅ REQUIRED: Use the exact filename from the mapping above

🎯 RANKING MISSION:
You are ranking {len(images)} water well images for maximum donor appeal.
Assign priorities from 1 to {len(images)} where:
- Priority {len(images)} = BEST image (will become _1_ in final naming)
- Priority {len(images)-1} = SECOND BEST image (will become _2_ in final naming)
- Priority 1 = WORST image (will become _{len(images)}_ in final naming)

🏆 TOP PRIORITY CRITERIA (for highest rankings):
1. **Priority {len(images)} (Future _1_)**: 
   - Donor plaque is clearly visible AND readable
   - Children are present, smiling, and engaged around the plaque
   - Natural joy and authentic expressions
   - Good lighting and composition
   - NEVER select plaque-only images for top priority

2. **Priority {len(images)-1} (Future _2_)**:
   - Active water interaction (children playing, drinking, collecting water)
   - Visible joy and excitement about water access
   - Clear action shots showing water flow
   - Engaging composition with happy subjects

📉 LOWER PRIORITY INDICATORS:
- Static poses or forced smiles
- Blurry or poorly lit images
- Plaque unreadable or partially obscured
- No visible water interaction
- Boring or repetitive compositions
- Technical issues (overexposed, underexposed, motion blur)

🔒 OUTPUT FORMAT REQUIREMENTS:
Return ONLY a valid JSON array with this exact structure:

[
  {{
    "id": "exact_image_id_from_input",
    "filename": "EXACT_FILENAME_FROM_INPUT_LIST",
    "priority": INTEGER_FROM_1_TO_{len(images)},
    "reason": "Brief justification (max 50 chars)"
  }}
]

🛡️ VALIDATION CHECKLIST:
- Each filename exists in input list: {list(input_filenames.keys())}
- All priorities 1-{len(images)} used exactly once
- No explanatory text outside JSON array
- All required fields present for each image

Water well context: {water_well_name or 'Unknown location'}
{f"Historical context: {history_context}" if history_context else ""}

RESPOND WITH ONLY THE JSON ARRAY - NO OTHER TEXT.
"""

    # Build Vision API request
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

        # Enhanced validation - check against the mapping
        if not isinstance(result, list) or len(result) != len(images):
            return {"error": f"Expected {len(images)} results, got {len(result)}"}

        # Validate priorities are unique and complete
        priorities = [item.get("priority") for item in result]
        expected_priorities = list(range(1, len(images) + 1))
        if sorted(priorities) != expected_priorities:
            return {"error": "Priority numbers missing or duplicated", 
                   "expected": expected_priorities, "received": sorted(priorities)}

        # CRITICAL: Validate filenames and IDs match input mapping exactly
        for item in result:
            returned_filename = item.get("filename", "")
            returned_id = item.get("id", "")
            
            # Find if this filename exists in our input mapping
            filename_found = False
            for img_key, img_data in input_images_map.items():
                if img_data['original_name'] == returned_filename and img_data['id'] == returned_id:
                    filename_found = True
                    break
            
            if not filename_found:
                return {"error": f"Vision Agent returned invalid filename/id combo: filename='{returned_filename}', id='{returned_id}'", 
                       "expected_mapping": input_images_map,
                       "returned_items": result}

        # Validate required fields
        for item in result:
            required_fields = ["id", "filename", "priority", "reason"]
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                return {"error": f"Missing required fields: {missing_fields}"}

        return result
        
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing failed: {e}", "raw_response": text}
    except Exception as e:
        return {"error": f"Result processing failed: {e}"}
