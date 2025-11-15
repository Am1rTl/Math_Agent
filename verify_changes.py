import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file:///app/verification.html")
        await page.screenshot(path="verification.png")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())