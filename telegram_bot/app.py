import asyncio
import logging
import yaml
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Load configuration from YAML file
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load variables from config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FASTAPI_URL = config["fastapi_url"]

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Enable logging
logging.basicConfig(level=logging.INFO)


@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Отправь мне текст, и я найду юридическую информацию.")


@dp.message()
async def handle_message(message: Message):
    user_text = message.text.strip()

    if not user_text:
        await message.answer("Пожалуйста, отправь текстовое сообщение.")
        return

    # await message.answer("🔍 Обрабатываю ваш запрос...")

    # Send request to FastAPI
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FASTAPI_URL, params={"query": user_text}) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("response", "❌ Ошибка: ответ не получен.")
                else:
                    answer = f"❌ Ошибка {response.status}: {await response.text()}"
        
        await message.answer(f"✅ Ответ:\n{answer}")

    except Exception as e:
        logging.error(f"Ошибка при запросе к FastAPI: {e}")
        await message.answer("❌ Произошла ошибка при обработке запроса. Попробуйте позже.")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
