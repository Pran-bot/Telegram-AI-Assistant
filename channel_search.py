import os
import re
import requests
import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
from groq import Groq

class ChannelSearch:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        if not self.api_key:
            raise ValueError("Missing SERPAPI_KEY environment variable")
        
        self.base_url = "https://serpapi.com/search.json"
        self.blacklist = [
            "t.me/joinchat", 
            "telegram.dog",
            "t-me.io",
            "telegram.scam"
        ]

    def clean_channel_name(self, url: str) -> str:
        """Extract channel name from URL"""
        name = url.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9]', ' ', name).title()
    
    def is_valid_channel(self, url: str) -> bool:
        """Check if URL is a valid Telegram channel"""
        return (url.startswith(('https://t.me/', 'https://telegram.me/')) 
                and not any(bad in url for bad in self.blacklist))

    async def generate_search_response(self, query: str, channels: List[Dict] = None) -> str:
        """Generate AI response about the search results"""
        try:
            if not channels:
                return f"I couldn't find any Telegram channels for '{query}'. Try different keywords or check back later."

            response = self.groq_client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You help users find Telegram channels. Be concise and friendly."
                }, {
                    "role": "user",
                    "content": f"I searched for '{query}' on Telegram. Generate a helpful response about these channels: {channels}"
                }],
                model="llama3-70b-8192",
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"AI response generation failed: {str(e)}")
            channel_names = [ch['name'] for ch in channels]
            return f"I found these Telegram channels for '{query}': {', '.join(channel_names)}"

    async def search_telegram_channels(self, query: str) -> JSONResponse:
        """Search using SerpAPI and return formatted results"""
        params = {
            "q": f"{query} download site:t.me",
            "api_key": self.api_key,
            "num": 5,
            "hl": "en"
        }
        
        try:
            # Make the API request
            response = requests.get(
                self.base_url, 
                params=params,
                timeout=10
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON safely
            try:
                data = response.json()
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid JSON response from search provider"
                )
            
            # Process results
            channels = []
            if "organic_results" in data:
                for result in data["organic_results"]:
                    url = result.get("link", "")
                    if self.is_valid_channel(url):
                        channels.append({
                            "name": self.clean_channel_name(url),
                            "url": url,
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", "")
                        })
                        if len(channels) >= 3:
                            break
            
            # Generate AI response
            ai_response = await self.generate_search_response(query, channels)
            
            # Return properly formatted JSON response
            return JSONResponse({
                "status": "success",
                "query": query,
                "response": ai_response,
                "channels": channels
            })
            
        except requests.exceptions.HTTPError as e:
            error_msg = "Search service is currently unavailable"
            if response.status_code == 429:
                error_msg = "Daily search limit reached. Please try again tomorrow."
            return JSONResponse(
                {"status": "error", "detail": error_msg},
                status_code=500
            )
        except requests.exceptions.Timeout:
            return JSONResponse(
                {"status": "error", "detail": "Search timed out. Please try again."},
                status_code=504
            )
        except Exception as e:
            return JSONResponse(
                {"status": "error", "detail": f"Search failed: {str(e)}"},
                status_code=500
            )

channel_searcher = ChannelSearch()