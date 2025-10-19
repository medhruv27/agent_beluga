"""
Web Scraper Agent for fetching latest GenAI updates and information
"""
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import logging
from datetime import datetime, timedelta
import json
from config import settings

logger = logging.getLogger(__name__)


class WebScraperAgent:
    """Handles web scraping for latest GenAI information"""
    
    def __init__(self):
        self.session = None
        self.genai_sources = [
            "https://blog.langchain.com/",
            "https://www.llamaindex.ai/blog",
            "https://huggingface.co/blog",
            "https://openai.com/blog",
            "https://ai.googleblog.com/",
            "https://www.anthropic.com/news",
            "https://mistral.ai/news/",
            "https://www.cohere.com/blog"
        ]
        
        self.research_sources = [
            "https://arxiv.org/list/cs.AI/recent",
            "https://arxiv.org/list/cs.CL/recent", 
            "https://arxiv.org/list/cs.CV/recent",
            "https://arxiv.org/list/cs.LG/recent"
        ]
        
        self.news_sources = [
            "https://venturebeat.com/ai/",
            "https://techcrunch.com/category/artificial-intelligence/",
            "https://www.theverge.com/ai-artificial-intelligence"
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_latest_updates(self, max_articles: int = 10) -> List[Dict[str, Any]]:
        """Scrape latest GenAI updates from various sources"""
        try:
            all_articles = []
            
            # Scrape from different source types
            tasks = [
                self._scrape_blog_posts(max_articles // 3),
                self._scrape_research_papers(max_articles // 3),
                self._scrape_news_articles(max_articles // 3)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in scraping task: {result}")
            
            # Sort by date and return most recent
            all_articles.sort(key=lambda x: x.get('date', datetime.min), reverse=True)
            return all_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error scraping latest updates: {e}")
            return []
    
    async def _scrape_blog_posts(self, max_posts: int) -> List[Dict[str, Any]]:
        """Scrape blog posts from GenAI company blogs"""
        posts = []
        
        for source in self.genai_sources[:3]:  # Limit to first 3 sources
            try:
                source_posts = await self._scrape_blog_source(source, max_posts // 3)
                posts.extend(source_posts)
            except Exception as e:
                logger.error(f"Error scraping {source}: {e}")
        
        return posts
    
    async def _scrape_blog_source(self, url: str, max_posts: int) -> List[Dict[str, Any]]:
        """Scrape a specific blog source"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    posts = []
                    # Generic selectors - would need to be customized per site
                    article_links = soup.find_all('a', href=True)[:max_posts]
                    
                    for link in article_links:
                        href = link.get('href')
                        if href and ('blog' in href or 'post' in href or 'article' in href):
                            title = link.get_text(strip=True)
                            if title and len(title) > 10:  # Filter out short titles
                                posts.append({
                                    'title': title,
                                    'url': href if href.startswith('http') else f"{url.rstrip('/')}/{href.lstrip('/')}",
                                    'source': url,
                                    'type': 'blog_post',
                                    'date': datetime.now() - timedelta(days=1)  # Placeholder
                                })
                    
                    return posts
                    
        except Exception as e:
            logger.error(f"Error scraping blog source {url}: {e}")
            return []
    
    async def _scrape_research_papers(self, max_papers: int) -> List[Dict[str, Any]]:
        """Scrape recent research papers from arXiv"""
        papers = []
        
        try:
            # Scrape arXiv recent papers
            for source in self.research_sources[:2]:  # Limit to first 2 sources
                source_papers = await self._scrape_arxiv_source(source, max_papers // 2)
                papers.extend(source_papers)
                
        except Exception as e:
            logger.error(f"Error scraping research papers: {e}")
        
        return papers
    
    async def _scrape_arxiv_source(self, url: str, max_papers: int) -> List[Dict[str, Any]]:
        """Scrape arXiv source for recent papers"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    papers = []
                    # arXiv specific selectors
                    paper_links = soup.find_all('a', href=True)[:max_papers]
                    
                    for link in paper_links:
                        href = link.get('href')
                        if href and 'abs/' in href:
                            title = link.get_text(strip=True)
                            if title and len(title) > 10:
                                papers.append({
                                    'title': title,
                                    'url': f"https://arxiv.org{href}",
                                    'source': 'arXiv',
                                    'type': 'research_paper',
                                    'date': datetime.now() - timedelta(days=1)  # Placeholder
                                })
                    
                    return papers
                    
        except Exception as e:
            logger.error(f"Error scraping arXiv source {url}: {e}")
            return []
    
    async def _scrape_news_articles(self, max_articles: int) -> List[Dict[str, Any]]:
        """Scrape news articles about GenAI"""
        articles = []
        
        try:
            for source in self.news_sources[:2]:  # Limit to first 2 sources
                source_articles = await self._scrape_news_source(source, max_articles // 2)
                articles.extend(source_articles)
                
        except Exception as e:
            logger.error(f"Error scraping news articles: {e}")
        
        return articles
    
    async def _scrape_news_source(self, url: str, max_articles: int) -> List[Dict[str, Any]]:
        """Scrape a specific news source"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    articles = []
                    # Generic news selectors
                    article_links = soup.find_all('a', href=True)[:max_articles]
                    
                    for link in article_links:
                        href = link.get('href')
                        title = link.get_text(strip=True)
                        if title and len(title) > 10 and href:
                            articles.append({
                                'title': title,
                                'url': href if href.startswith('http') else f"{url.rstrip('/')}/{href.lstrip('/')}",
                                'source': url,
                                'type': 'news_article',
                                'date': datetime.now() - timedelta(days=1)  # Placeholder
                            })
                    
                    return articles
                    
        except Exception as e:
            logger.error(f"Error scraping news source {url}: {e}")
            return []
    
    async def scrape_article_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape full content of a specific article"""
        try:
            async with self.session.get(url, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract main content
                    content_selectors = [
                        'article', 'main', '.content', '.post-content', 
                        '.article-content', '.entry-content'
                    ]
                    
                    content = ""
                    for selector in content_selectors:
                        element = soup.select_one(selector)
                        if element:
                            content = element.get_text(strip=True)
                            break
                    
                    if not content:
                        # Fallback to body text
                        content = soup.get_text(strip=True)
                    
                    # Extract metadata
                    title = soup.find('title')
                    title = title.get_text(strip=True) if title else "Untitled"
                    
                    return {
                        'title': title,
                        'content': content[:5000],  # Limit content length
                        'url': url,
                        'scraped_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {e}")
            return None
    
    async def search_genai_trends(self, query: str) -> List[Dict[str, Any]]:
        """Search for specific GenAI trends and topics"""
        try:
            # This would integrate with search APIs like SerpAPI
            # For now, return a placeholder implementation
            trends = []
            
            # Simulate trend search results
            trend_keywords = [
                "large language models", "multimodal AI", "RAG systems",
                "vector databases", "AI agents", "fine-tuning", "prompt engineering"
            ]
            
            for keyword in trend_keywords:
                if keyword.lower() in query.lower():
                    trends.append({
                        'keyword': keyword,
                        'relevance_score': 0.8,
                        'trend_data': {
                            'mentions': 150,
                            'growth_rate': 25,
                            'sources': ['GitHub', 'arXiv', 'Medium']
                        }
                    })
            
            return trends
            
        except Exception as e:
            logger.error(f"Error searching GenAI trends: {e}")
            return []
    
    async def get_market_updates(self) -> List[Dict[str, Any]]:
        """Get latest market updates and funding news"""
        try:
            # This would integrate with financial APIs
            # For now, return placeholder data
            updates = [
                {
                    'company': 'OpenAI',
                    'update': 'Latest funding round and valuation',
                    'date': datetime.now() - timedelta(days=1),
                    'type': 'funding',
                    'relevance': 'high'
                },
                {
                    'company': 'Anthropic',
                    'update': 'New model release and capabilities',
                    'date': datetime.now() - timedelta(days=2),
                    'type': 'product',
                    'relevance': 'high'
                }
            ]
            
            return updates
            
        except Exception as e:
            logger.error(f"Error getting market updates: {e}")
            return []
