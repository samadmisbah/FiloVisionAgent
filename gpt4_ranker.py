async def rank_images(images, history_folder=None, water_well_name=None, max_selections=10, **kwargs):
    if not images:
        return {"error": "No valid images provided"}

    history_context = ""
    if history_folder and GOOGLE_API_AVAILABLE:
        folder_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', history_folder)
        if folder_match:
            folder_id = folder_match.group(1)
            history_context = await get_history_examples(folder_id)

    # CREATE THE FILENAME MAPPING - This is what was missing!
    image_mapping = []
    for i, img in enumerate(images):
        filename = img.get('name', img.get('filename', f'image_{i}.jpg'))
        image_mapping.append(f"Image {i+1}: filename='{filename}', id='{img.get('id', '')}'")
    
    mapping_text = "\n".join(image_mapping)

    prompt = f"""
You will ONLY return a strict JSON array output — no explanation, no preamble.

You are ranking {len(images)} water well images for donor appeal.
Assign a unique `priority` from 1 (worst) to {len(images)} (best), using each number once.

🎯 GOAL: Identify the **top 2 donor images**:
- `_1_`: clearest plaque WITH children who are smiling, happy, or joyful — ideally around the plaque. Do NOT select a plaque-only image even if it's the clearest plaque. There must be joyful children around it for _1_.
- `_2_`: joyful interaction with water (children splashing, smiling, visibly enjoying)

🧠 RANKING RULES:
✅ Rank highest:
- Children directly playing with water or holding containers
- Plaque is readable and framed well WITH children smiling
- Natural joy and expressive faces
- Clean background, clear lighting

❌ Rank lower (3–10):
- No water flow or joy
- Faces are turned, bored, or unclear
- Plaque cut off or out of frame
- Plaque only without children should never be _1_
- Crowded, blurry, redundant, or awkward composition

🧪 Priority Definitions:
{len(images)} → Best image for `_1_`
{len(images)-1} → Best image for `_2_`
1 → Worst image in batch (static, joyless, poor visibility)

📎 CRITICAL FILENAME MAPPING - USE THESE EXACT FILENAMES:
{mapping_text}

❌ NEVER use generic names like "image1.jpg", "image2.jpg" 
✅ ONLY use the exact filenames listed above

Respond with ONLY a valid JSON array. Do not explain anything else.

[
  {{
    "id": "exact-id-from-mapping-above",
    "filename": "exact-filename-from-mapping-above",
    "priority": 1,
    "reason": "Short visual justification"
  }},
  ...
]
"""

    # Rest of the function remains exactly the same...
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

        # Validate unique priorities
        priorities = [item.get("priority") for item in result]
        expected = list(range(1, len(images) + 1))
        if sorted(priorities) != expected:
            return {"error": "Priority numbers missing or duplicated", "priorities": priorities}

        # ADD SIMPLE FILENAME VALIDATION
        input_filenames = [img.get('name', img.get('filename', '')) for img in images]
        for item in result:
            returned_filename = item.get("filename", "")
            if returned_filename not in input_filenames:
                return {"error": f"Invalid filename returned: '{returned_filename}'. Must use one of: {input_filenames}"}

        return result
    except Exception as e:
        return {"error": f"Result parsing failed: {e}"}
