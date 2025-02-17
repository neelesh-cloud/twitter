from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from transformers import pipeline
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging
import traceback
import threading
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RequestData(BaseModel):
    text: str

class TwitterSentimentScraper:
    def __init__(self, headless=True, sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initializes WebDriver, Sentiment Analysis Pipeline, and Translator."""
        try:
            self.geckodriver_path = os.getenv("GECKODRIVER_PATH", "/Users/neeleshpandya/Desktop/tweeter/xcom/geckodriver")
            options = Options()
            if headless:
                options.add_argument("--headless")

            # Initialize WebDriver
            self.service = Service(self.geckodriver_path)
            self.driver = webdriver.Firefox(service=self.service, options=options)
            self.wait = WebDriverWait(self.driver, 5)

            logger.info("WebDriver initialized successfully.")

            # Initialize Sentiment Analysis Model
            self.sentiment_pipe = pipeline("text-classification", model=sentiment_model)
            self.translator = Translator()

            # ThreadPool for parallel sentiment analysis
            self.executor = ThreadPoolExecutor(max_workers=4)

        except WebDriverException as e:
            logger.error(f"Error initializing WebDriver: {e}")
            logger.debug(traceback.format_exc())
            raise

    def analyze_sentiment_async(self, text):
        """Runs sentiment analysis in a separate thread."""
        return self.sentiment_pipe(text)[0]

    def scrape_tweets(self, query, num_pages=2):
        """Scrapes tweets from Nitter and performs sentiment analysis in parallel."""
        url = f"https://nitter.net/search?f=tweets&q={query}"
        logger.info(f"Starting scrape for query: {query}, Number of pages: {num_pages}")
        
        try:
            logger.info(f"Accessing URL: {url}")
            self.driver.get(url)
             # Increase the timeout to 30 seconds (you can adjust this as needed)
            self.wait = WebDriverWait(self.driver, 5)  # Set wait timeout to 30 seconds
            # Adding a longer timeout, if needed. Adjust the time (e.g., 20 seconds) as required.
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "timeline")))
            
            logger.info(f"Successfully accessed and found tweets on URL: {url}")
        except TimeoutException as e:
            logger.warning(f"TimeoutException: Failed to load tweets within the expected time. URL: {url}")
            logger.debug(f"Detailed Error: {str(e)}")
            return []
        except WebDriverException as e:
            logger.error(f"WebDriverException: Error loading URL {url}. Error Details: {str(e)}")
            logger.debug(f"Stack Trace: {traceback.format_exc()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error occurred while scraping: {str(e)}")
            logger.debug(f"Stack Trace: {traceback.format_exc()}")
            return []

        tweet_data = []

        def extract_tweets():
            """Extracts tweets from the page and performs sentiment analysis."""
            try:
                tweets = self.driver.find_elements(By.CLASS_NAME, "timeline-item")
                if not tweets:
                    logger.warning("No tweets found on the page.")

                for tweet in tweets:
                    try:
                        tweet_link = tweet.find_element(By.CLASS_NAME, "tweet-link").get_attribute("href")
                        username = tweet.find_element(By.CLASS_NAME, "username").text.strip()
                        full_name = tweet.find_element(By.CLASS_NAME, "fullname").text.strip()
                        tweet_date = tweet.find_element(By.CLASS_NAME, "tweet-date").text.strip()
                        tweet_content = tweet.find_element(By.CLASS_NAME, "tweet-content").text.strip()

                        stats = tweet.find_elements(By.CLASS_NAME, "tweet-stat")
                        comments, retweets, quotes, likes = (
                            stats[i].text.strip() if i < len(stats) else "0" for i in range(4)
                        )

                        try:
                            image_url = tweet.find_element(By.CLASS_NAME, "avatar.round").get_attribute("src")
                        except NoSuchElementException:
                            image_url = None

                        if not tweet_content.isascii():
                            try:
                                trans_content = self.translator.translate(tweet_content, src='auto', dest='en').text
                            except Exception as e:
                                logger.error(f"Error translating tweet: {e}")
                                trans_content = tweet_content
                        else:
                            trans_content = tweet_content

                        future = self.executor.submit(self.analyze_sentiment_async, trans_content)
                        sentiment_result = future.result()
                        sentiment = sentiment_result["label"]
                        sentiment_score = sentiment_result["score"]

                        tweet_data.append({
                            "tweet_url": tweet_link,
                            "username": username,
                            "full_name": full_name,
                            "tweet_date": tweet_date,
                            "tweet_content": tweet_content,
                            "trans_content": trans_content,
                            "sentiment": sentiment,
                            "sentiment_score": sentiment_score,
                            "comments": comments,
                            "retweets": retweets,
                            "quotes": quotes,
                            "likes": likes,
                            "image_url": image_url
                        })
                    except Exception as e:
                        logger.error(f"Error extracting tweet: {e}")
                        logger.debug(traceback.format_exc())

            except Exception as e:
                logger.error(f"Error while extracting tweets: {e}")
                logger.debug(traceback.format_exc())

        extract_tweets()

        logger.info("Twitter data scraping started...")

        for _ in range(num_pages):
            try:
                load_more_button = self.driver.find_element(By.CLASS_NAME, "show-more")
                self.driver.execute_script("arguments[0].click();", load_more_button)
                time.sleep(2)
                extract_tweets()
            except NoSuchElementException:
                logger.warning("No more 'Load More' button found. Stopping pagination.")
                break
            except Exception as e:
                logger.error(f"Error clicking 'Load More' button: {e}")
                logger.debug(traceback.format_exc())
                break

        logger.info(f"Twitter data scraping completed. {len(tweet_data)} tweets extracted.")

        return tweet_data

    def close(self):
        """Closes the WebDriver."""
        try:
            self.driver.quit()
            self.executor.shutdown(wait=False)
            logger.info("WebDriver and executor closed successfully.")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")
            logger.debug(traceback.format_exc())

# Global scraper instance and lock
scraper_lock = threading.Lock()
scraper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of resources."""
    global scraper
    logger.info("Starting FastAPI application...")
    with scraper_lock:
        if scraper is None:
            scraper = TwitterSentimentScraper()
    yield  # Application runs while this is active
    if scraper is not None:
        scraper.close()
    logger.info("Shutting down FastAPI application...")

app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def get_scraper():
    """Dependency to get scraper instance."""
    global scraper
    with scraper_lock:
        if scraper is None:
            scraper = TwitterSentimentScraper()
    return scraper

@app.post("/analyze")
def analyze(request: RequestData, scraper_instance=Depends(get_scraper)):
    """API endpoint to analyze tweets for sentiment."""
    try:
        result = scraper_instance.scrape_tweets(request.text)
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
