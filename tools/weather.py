import httpx

# Tool 定义，供 OpenAI function calling 使用
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，支持中文或英文，例如：北京、Shanghai",
                }
            },
            "required": ["city"],
        },
    },
}


async def get_weather(city: str) -> str:
    """调用 wttr.in 获取天气，无需 API key"""
    url = f"https://wttr.in/{city}?format=j1&lang=zh"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        area_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        desc = current["lang_zh"][0]["value"] if current.get("lang_zh") else current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_speed = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]

        return (
            f"{area_name}, {country} 当前天气：{desc}，"
            f"气温 {temp_c}°C（体感 {feels_like}°C），"
            f"湿度 {humidity}%，风速 {wind_speed} km/h（{wind_dir}）"
        )
    except Exception as e:
        return f"获取 {city} 天气失败：{e}"
