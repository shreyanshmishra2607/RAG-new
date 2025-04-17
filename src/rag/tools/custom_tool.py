from crewai_tools import ScrapeWebsiteTool

def extract_topic_from_url(url):
    """
    Extract topic from the last part of the URL.
    """
    return url.rstrip('/').split('/')[-1].replace('-', ' ').replace('_', ' ').title()

class ScrapeAndProcessTool:
    def __init__(self, urls, user_prompt):
        self.urls = urls
        self.user_prompt = user_prompt

    def scrape_and_process(self):
        scrape_tool = ScrapeWebsiteTool()
        scraped_text = ""
        
        topics = []
        
        # Scrape each URL and extract topics
        for url in self.urls:
            scrape_tool = ScrapeWebsiteTool(website_url=url)
            scraped_text += scrape_tool.run() + "\n"
            topics.append(extract_topic_from_url(url))
        
        # Merge extracted topics into one
        merged_topic = ' '.join(topics)

        # Return the merged topic, scraped content, and user prompt
        return merged_topic, scraped_text, self.user_prompt
