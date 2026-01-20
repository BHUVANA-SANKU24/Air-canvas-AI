import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


class WebQA:
    def __init__(self, timeout=10):
        self.timeout = timeout

        # Question templates for different keyword types
        self.question_templates = {
            "who": [
                "Who is {keyword}?",
                "Who was {keyword}?",
                "Who are {keyword}?"
            ],
            "what": [
                "What is {keyword}?",
                "What are {keyword}?",
                "What does {keyword} mean?"
            ],
            "where": [
                "Where is {keyword}?",
                "Where are {keyword} located?",
                "Where can I find {keyword}?"
            ],
            "when": [
                "When did {keyword} happen?",
                "When was {keyword}?",
                "When is {keyword}?"
            ],
            "how": [
                "How does {keyword} work?",
                "How to {keyword}?",
                "How is {keyword} done?"
            ],
            "why": [
                "Why is {keyword} important?",
                "Why does {keyword} happen?",
                "Why {keyword}?"
            ],
            "general": [
                "Tell me about {keyword}",
                "What is {keyword}?",
                "Information about {keyword}"
            ]
        }

    def clean_text(self, s: str) -> str:
        """Clean and normalize text"""
        if not s:
            return ""
        s = s.replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # Remove junk
        s = re.sub(
            r"(cookie|privacy policy|terms of use|subscribe|sign in|advertisement)",
            "", s, flags=re.I
        )
        return s.strip()

    def generate_question(self, keyword: str) -> str:
        """
        Generate a proper question from a keyword or short phrase
        """
        keyword = self.clean_text(keyword).lower()

        if not keyword:
            return None

        # If already a question, return as is
        if any(keyword.startswith(q) for q in ["who", "what", "where", "when", "why", "how"]):
            return keyword.capitalize()

        # Detect question type based on keyword
        words = keyword.split()

        # Check if it's a person's name (capitalized words)
        if len(words) <= 3 and all(w[0].isupper() for w in words if len(w) > 0):
            template = self.question_templates["who"][0]

        # Check for place/location keywords
        elif any(word in keyword for word in ["city", "country", "place", "location", "mountain", "river"]):
            template = self.question_templates["where"][0]

        # Check for time/date keywords
        elif any(word in keyword for word in ["date", "time", "year", "day", "event", "war", "independence"]):
            template = self.question_templates["when"][0]

        # Check for process/method keywords
        elif any(word in keyword for word in ["make", "create", "build", "cook", "work"]):
            template = self.question_templates["how"][0]

        # Check for reason keywords
        elif any(word in keyword for word in ["reason", "cause", "important", "matter"]):
            template = self.question_templates["why"][0]

        # Default to "what is"
        else:
            template = self.question_templates["what"][0]

        return template.format(keyword=keyword)

    def search(self, query, max_results=5):
        """Search using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []

    def fetch_text(self, url):
        """
        Scrape text from URL as fallback
        Extract only paragraph content
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            # Remove junk sections
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            # Get first few paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text(" ", strip=True)
                            for p in paragraphs[:8]])
            text = self.clean_text(text)

            return text
        except Exception as e:
            print(f"⚠️ Fetch error: {e}")
            return ""

    def extract_answer(self, text, max_sentences=3):
        """
        Extract concise answer from text
        Aim for 1-3 sentences
        """
        text = self.clean_text(text)
        if not text:
            return ""

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        good_sentences = []
        for s in sentences:
            s = s.strip()

            # Filter out junk sentences
            if len(s) < 20:
                continue
            if any(x in s.lower() for x in ["cookie", "privacy", "subscribe", "login", "click here"]):
                continue
            if s.count("?") > 2:  # Likely spam
                continue

            good_sentences.append(s)

            # Stop after getting enough content
            if len(good_sentences) >= max_sentences:
                break

        return " ".join(good_sentences) if good_sentences else ""

    def answer(self, user_input: str) -> tuple:
        """
        Main answer function
        Returns: (generated_question, answer)
        """
        user_input = self.clean_text(user_input)

        if len(user_input) < 2:
            return None, "Write at least one word."

        # Generate proper question
        question = self.generate_question(user_input)

        if not question:
            return None, "Could not understand the input."

        # Search for answer
        results = self.search(question)

        if not results:
            return question, "No web results found."

        # Try to get answer from search snippets first (fastest & cleanest)
        for r in results:
            snippet = r.get("body", "") or r.get("snippet", "")
            snippet = self.clean_text(snippet)

            if len(snippet) > 30:
                # Extract concise answer
                answer = self.extract_answer(snippet, max_sentences=3)
                if answer:
                    return question, answer

        # Fallback: scrape the first result
        url = results[0].get("href", "")
        if url:
            scraped = self.fetch_text(url)
            answer = self.extract_answer(scraped, max_sentences=3)

            if answer:
                return question, answer

        return question, "Could not extract a clear answer from web."
