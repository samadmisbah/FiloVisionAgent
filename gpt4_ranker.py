import aiohttp
import openai
import asyncio

openai.api_key = os.getenv("OPENAI_API_KEY")

async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                return None

async def rank_images(images, history_folder):
    tasks = [download_image(img["url"]) for img in images]
    raw_images = await asyncio.gather(*tasks)

    results = []
    for i, img_bytes in enumerate(raw_images):
        if not img_bytes:
            continue
        try:
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Rank this image for importance to a donor. Prefer clear plaques, smiling children, and emotional content."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_bytes.encode('base64')}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            caption = response.choices[0].message.content.strip()
            results.append({
                "filename": images[i]["name"],
                "priority": i + 1,
                "reason": caption,
                "id": images[i].get("id", "unknown")
            })
        except Exception as e:
            results.append({
                "filename": images[i]["name"],
                "priority": 999,
                "reason": f"Error analyzing image: {str(e)}",
                "id": images[i].get("id", "unknown")
            })
    return results
