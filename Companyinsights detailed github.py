import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
#from transformers import pipeline
from dotenv import load_dotenv
from qdrant_client import QdrantClient
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import requests
from bs4 import BeautifulSoup
import re
from google import genai

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import markdown2
from weasyprint import HTML
from urllib.parse import urlparse, urlunparse
from weasyprint import HTML, CSS
import markdown2
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import openai
load_dotenv()
import json
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import re
import io
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import qdrant_client
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
import urllib.parse

client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))
llm = client.models.get(model='gemini-2.0-flash-exp')

 #openai.api_key = Uncomment this line Give your OpenAPI key here, 

#qdrant_url = Uncomment this line, Give your vector quadrant url here
#qdrant_api_key = Uncomment this line,Give your vector quadrant API key here



class WebsiteScraper:
    def __init__(self, base_url, delay=1):
        """
        Initialize the scraper with a base URL and optional delay between requests
        
        Args:
            base_url (str): The main website URL to scrape
            delay (int): Delay in seconds between requests to avoid overwhelming the server
        """
        self.base_url = base_url
        self.delay = delay
        self.visited_urls = set()
        self.session = requests.Session()
        # Use a common User-Agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        parsed = urlparse(self.base_url)
        if not parsed.scheme:
            self.base_url =  urlunparse(('https', parsed.netloc or parsed.path, '', '', '', ''))
        self.companyname = self.get_company_name()
        self.garnerlink = self.get_gartner_review_link()
    def split_content(self,content, max_tokens=1000):
        #print(f"Inside split_content function: {content}")
        #print(type(content))
        if isinstance(content, dict):
            #print(content.values())
            content = str(content.values())  # Replace 'text' with actual key
            if not content:
                raise ValueError("Content dictionary does not contain valid text data.")
        
        elif not isinstance(content, str):
            raise TypeError("Content must be a string or a dictionary containing text.")
        

        #print(type(content))
        paragraphs = content.split('\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            if len(current_chunk + paragraph) < max_tokens:
                current_chunk += paragraph + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


    def extract_clean_text_from_pdf(self,url):
        try:
            # Fetch the PDF content from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Load the PDF content into PyPDF2
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        except requests.RequestException as e:
            print(f"Error fetching the PDF: {e}")
        except Exception as e:
            print(f"Error reading the PDF: {e}")

    def extract_clean_text_from_pdf1(self,pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            page_text = page.get_text()
            if page_text:  # Text-based page
                text += page_text + "\n"
            else:  # OCR for image-based pages
                images = page.get_images(full=True)
                for img in images:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    text += pytesseract.image_to_string(image) + "\n"
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        return text



    # Step 3: Summarize each chunk
    def summarize_chunk(self, chunk, max_tokens=2000):
        if not chunk.strip():
            raise ValueError("Chunk is empty or null.")
        #print(chunk)
        # OpenAI GPT
        #  response = openai.ChatCompletion.create(
        #    model="gpt-4",
        #    messages=[
        #        {"role": "system", "content": "You are a wonderful assistant, who is master in translating webscrapped content to natural language"},
        #        {"role": "user", "content": f"Summarize the following web scrapped content in natural language:\n\n{chunk}"}
        #    ],
        #    max_tokens=max_tokens,
        #    temperature=0.5
        #)
        #summary = response.choices[0].message.content.strip()

        # Google Gemini
        prompt = f"Summarize the following web scrapped content in natural language: {chunk}"
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config={'temperature': 0.1}
        )
        gemini_summary = response.candidates[0].content.parts[0].text.strip()

        return gemini_summary

    # Step 4: Final summary
    def summarize_web_content(self, content, max_tokens=1000, chunk_max_tokens=500, summary_max_tokens=2000):
        #print(content)
        chunks = self.split_content(content,max_tokens=max_tokens)
        #print("Chunks created:", len(chunks))

        summaries = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                print(f"Skipping empty chunk {i}.")
                continue
            #print(f"Summarizing chunk {i}...")
            summary = self.summarize_chunk(chunk)
            summaries.append(summary)
        
        final_summary = "\n".join(summaries)
        #print(f"Final website summary:\n{final_summary}")
        return final_summary


    def is_valid_url(self, url):
        """Check if URL belongs to the same domain as base_url"""
        base_domain = urlparse(self.base_url).netloc
        url_domain = urlparse(url).netloc
        return base_domain == url_domain

    def get_page_content(self, url):
        links = []
        if urlparse(url).path.lower().endswith('.pdf'):
            #print("this is pdf file")
            return self.extract_clean_text_from_pdf(url),[]

        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and img elements
            for tag in soup(["script", "style", "img"]):  # Added "img" to the tags to remove
                tag.decompose()
            
            # Get text content
            text_content = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'[^\x00-\x7F]+', ' ', text_content)  # Remove non-ASCII characters
            text = re.sub(r'[\r\n\t]+', ' ', text)      # Replace newlines and tabs with spaces
            text = re.sub(r'\s+', ' ', text)           # Collapse multiple spaces into one
           
            cleaned_text = ' '.join(text.split())
       
            
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self.is_valid_url(absolute_url):  # Assuming self.is_valid_url is defined
                    links.append(absolute_url)
                    
            return cleaned_text, links
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None, []
    def get_page_content1(self, url):
        """
        Get the text content and links from a webpage
        
        Args:
            url (str): URL to scrape
            
        Returns:
            tuple: (text_content, list_of_links)
        """
        try:
            # Respect robots.txt and add delay
            time.sleep(self.delay)
            
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text_content = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'[^\x00-\x7F]+', ' ', text_content)  # Remove non-ASCII characters
            text = re.sub(r'[\r\n\t]+', ' ', text)      # Replace newlines and tabs with spaces
            text_content = re.sub(r'\s+', ' ', text)           # Collapse multiple spaces into one
             

            
            # Find all links
            links = []
            for link in soup.find_all('a', href=True):  
                absolute_url = urljoin(url, link['href'])
                if self.is_valid_url(absolute_url):
                    links.append(absolute_url)
                    
            return text_content, links
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None, []

    def scrape_website(self, max_pages=None):
        """
        Scrape the website and its internal links
        
        Args:
            max_pages (int): Maximum number of pages to scrape (None for unlimited)
            
        Returns:
            dict: Dictionary with URLs as keys and their content as values
        """
        pages_content = {}
        urls_to_visit = [self.base_url]
        pages_scraped = 0
        
        while urls_to_visit and (max_pages is None or pages_scraped < max_pages):
        #while (pages_scraped == 0):
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            print(f"Scraping: {current_url}")

            content, links = self.get_page_content(current_url)
            if content:

                pages_content[current_url] = content
                self.visited_urls.add(current_url)
                pages_scraped += 1
                
                # Add new links to visit
                for link in links:
                    if link not in self.visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)
                        
        return pages_content
    def scrape_linkedin(self, max_pages=1):
        """
        Scrape the website and its internal links
        
        Args:
            max_pages (int): Maximum number of pages to scrape (None for unlimited)
            
        Returns:
            dict: Dictionary with URLs as keys and their content as values
        """
        pages_content = {}
        urls_to_visit = [self.find_linkedin_profile_1()]
        
        print(urls_to_visit)
        pages_scraped = 0
        print(f"Scraping: {urls_to_visit}")
        while urls_to_visit and (max_pages is None or pages_scraped < max_pages):
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            print(f"Scraping: {current_url}")
                
            content, links = self.get_page_content(current_url)
            
            if content:
                pages_content[current_url] = content
                self.visited_urls.add(current_url)
                pages_scraped += 1
                
                # Add new links to visit
                for link in links:
                    if link not in self.visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)
                        
        return pages_content
    def scrape_gartner(self, max_pages=1):
        """
        Scrape the website and its internal links
        
        Args:
            max_pages (int): Maximum number of pages to scrape (None for unlimited)
            
        Returns:
            dict: Dictionary with URLs as keys and their content as values
        """
        pages_content = {}
        urls_to_visit = [self.get_gartner_review_link()]
        
        print(urls_to_visit)
        pages_scraped = 0
        print(f"Scraping: {urls_to_visit}")
        while urls_to_visit and (max_pages is None or pages_scraped < max_pages):
        #while (pages_scraped == 0):
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            print(f"Scraping: {current_url}")
                
            content, links = self.get_page_content(current_url)
            
            if content:
                pages_content[current_url] = content
                self.visited_urls.add(current_url)
                pages_scraped += 1
                
                # Add new links to visit
                for link in links:
                    if link not in self.visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)
                        
        return pages_content
    def get_linkedin_profile(self):
        query = f"site:linkedin.com/company {self.base_url}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        # Perform Google Search
        url = f"https://www.google.com/search?q={query}"
        response = requests.get(url, headers=headers)
        
        # Check for a successful response
        if response.status_code != 200:
            return "Error in Google search request"
        
        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all links in the search results
        for link in soup.find_all("a"):
            href = link.get("href")
            
            # Only process URLs that contain "linkedin.com/company"
            if href and "linkedin.com/company" in href:
                # Clean up the URL to get the direct LinkedIn profile URL
                linkedin_url = re.search(r"(https://[a-z]{2,3}\.linkedin\.com/company/[^\s]+)", href)
                
                # If we find the URL, return it
                if linkedin_url:
                    return linkedin_url.group(1)
        
        return "LinkedIn profile not found."

    def find_linkedin_profile_1(self):
        try:
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all anchor tags with href attributes
            links = soup.find_all('a', href=True)

            # Search for LinkedIn URL
            for link in links:
                if 'linkedin.com/company/' in link['href']:
                    return link['href']
            return "LinkedIn profile not found"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def find_linkedin_profile_2(self):
        prompt =f" Just give the link of Linkedin page of {self.base_url}. No other text or other text or symbols to be included"
        response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt)
        return  response.candidates.pop(0).content.parts.pop(0).text
    def save_content_to_file(content_dict, filename="website_content.txt"):
        """Save the scraped content to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for url, content in content_dict.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"URL: {url}\n")
                f.write(f"{'='*80}\n\n")
                f.write(content)
                f.write("\n\n")

    def get_competitors(self):
        prompt =f" Please give list of 2 closest relevant competitor's website for this site: {self.base_url}. No other remarks or other text or symbols to be included"
        response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt)
        input_string = response.candidates.pop(0).content.parts.pop(0).text
        # Regular expression to match URLs
        url_pattern = r'(https?://[^\s,]+|www\.[^\s,]+)'
        # Extract websites into a list
        Competitor_websites = re.findall(url_pattern, input_string)
        return Competitor_websites


    def get_company_name(self):
        prompt =f"Just give the company name from the company's website: {self.base_url}. No tags or special characters please."
       

        models = openai.models.list()

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helper to a marketing person."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Limit the response length
            temperature=0.7  # Controls randomness: 0 is deterministic, 1 is more random
        )

        companyname = response.choices.__getitem__(0).message.content

        print(companyname)
        return companyname

    def get_gartner_review_link(self):
        search_url = f"https://www.gartner.com/en/search?q={self.companyname}"
            
        headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
        response = requests.get(search_url, headers=headers)
            
        if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                # Continue with scraping logic here
                print("Page retrieved successfully!")
        else:
                print(f"Failed to retrieve search results: {response.status_code}")

        prompt =f"Just give gartner link of the company's website: {self.companyname}. No tags or special characters please."
       

        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helper to a marketing person."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Limit the response length
            temperature=0.7  # Controls randomness: 0 is deterministic, 1 is more random
        )
        #companyname =  response.candidates.pop(0).content.parts.pop(0).text
        gartnerlink = response.choices.__getitem__(0).message.content
        return gartnerlink


    def get_total_company_info(self):
        scraper = WebsiteScraper(
            base_url= self.base_url,  # Replace with your target website

            delay=1  # 1 second delay between requests
        )
        # Scrape the website (limit to 10 pages for this example)
        content_website = scraper.scrape_website(max_pages=10)

        content_linkedin = scraper.scrape_linkedin(max_pages=3)
    
        prompt =f"Translate this webscrapped content {content_website} to Natural Language"
        content_website = self.summarize_web_content(content_website)

        prompt =f"Give detailed company insights about this company which has this website: {self.base_url}."
        
        response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt,config={'temperature': 0.1})
        content_LLM = response.candidates.pop(0).content.parts.pop(0).text
        
        #print(content_LLM)
        prompt =f"Translate this linkedin scrapped content  to natural language: {content_linkedin} - seggregate this clearly that this information is got from linkedin site, and create overall writeup consolidating all information."
        
        response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt,config={'temperature': 0.1})
        content_linkedin = response.candidates.pop(0).content.parts.pop(0).text

           #print(content_LLM)
        prompt =f"Translate this website scrapped content  to natural language: {content_website} - seggregate this clearly that this information is got from web site, and create overall writeup consolidating all information."
        
        response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt,config={'temperature': 0.1})
        content_website = response.candidates.pop(0).content.parts.pop(0).text

        prompt =f" For the company which has this website: {self.base_url}, details generated from LLM is   is {content_LLM}, details generated from linkedin is {content_linkedin} ,details generated from  website page is {content_website}.  Now consolidate all inputs. Make sure you dont miss any information from all the inputs given. "
        response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt,config={'temperature': 0.1})
        total_contents = response.candidates.pop(0).content.parts.pop(0).text
  
        outstr = " \n\n Summarized information is : \n\n " + str(total_contents)
        with open(f"{self.companyname}.md", "w") as file:
           
            file.write(outstr)

         # Convert Markdown to HTML
        html_string = markdown2.markdown(outstr)
        
        # Convert the HTML to PDF
        HTML(string=html_string).write_pdf(f"{self.companyname}.pdf")
        return total_contents


# Example usage
if __name__ == "__main__":
    # Initialize the scraper

  
    scraper = WebsiteScraper(
        
        base_url = , #Give your company website here, whichever you want
       
        delay=1  # 1 second delay between requests
    )
    companyname = scraper.get_company_name()
    gart = scraper.get_gartner_review_link()
    print(gart)
    main_companyinfo = scraper.get_total_company_info()
    quadrantString = main_companyinfo
    competitor_info = f"\n\n Now let us look at the details of the competitor companies of {companyname} \n\n"
    competitors = scraper.get_competitors()
    for item in competitors:
        competitor_info = competitor_info + f"\n\n Details about the competitor : {item}: \n\n"
        scraper_c = WebsiteScraper(
            base_url=item,  # Replace with your target website
            delay=1  # 1 second delay between requests
        )
        cn = scraper_c.companyname
        #print(len(competitor_info))
        cc = scraper_c.get_total_company_info()
        #prompt =f"For the company - {item}, we have website scrapped content about the company here: {cc}. Make sure you dont miss any of the information specified in the input data"
        #response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt, config={'temperature': 0.1})
        #cc = response.candidates.pop(0).content.parts.pop(0).text
        html_string = markdown2.markdown(cc)
        
        # Convert the HTML to PDF
        HTML(string=html_string).write_pdf(f"CompetitorCompany_{cn}_detailed.pdf")
        competitor_info = competitor_info + cc

    #print(competitors)
    #print(competitor_info)
    quadrantString = quadrantString + competitor_info

    #print(companyname)
    prompt =f"For the company - {companyname}, we have detailed writeup about the company here: {main_companyinfo}. Details about their competitors are given here: {competitor_info}. We have to  extracting complete information on the company-   {companyname}, also compare the company against their competitors.  Also create a table of comparison. Table alone should be in markup language, and other text should be in natural language. Make sure you dont miss any of the information specified in the input data"
    
    response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt, config={'temperature': 0.1})
    detailed_company_report = response.candidates.pop(0).content.parts.pop(0).text
    #print(detailed_company_report)
    quadrantString = quadrantString + detailed_company_report
     # Convert Markdown to HTML
    overall_output = detailed_company_report +  f"\n\n Details about the company {companyname}  is : \n\n" + main_companyinfo + "\n\n Details about competitors companies is  \n\n" + competitor_info
    html_string = markdown2.markdown(detailed_company_report)
        
        # Convert the HTML to PDF
    HTML(string=html_string).write_pdf(f"{companyname}_detailed.pdf")

    html_string = markdown2.markdown(overall_output)
        
        # Convert the HTML to PDF
    HTML(string=html_string).write_pdf(f"{companyname}_detailed_withcompetitorInfo.pdf")
    


    comparisonMetrics = """1. Financial Performance which covers Revenue (Total/Annual/Quarterly) , Net Income or Profit Margin, Operating Expenses
EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
Market Capitalization
Return on Investment (ROI) or Return on Equity (ROE)
2. Market and Customer Metrics
Market Share
Customer Base (e.g., Number of Active Customers)
Customer Retention Rate
Customer Satisfaction (e.g., NPS - Net Promoter Score)
3. Product and Service Offerings
Product Diversity (Number and Variety of Products)
Service Quality or Innovation
Product Lifecycle (e.g., Mature vs. Emerging Products)
Differentiators (Unique Selling Points)
4. Operational Metrics
Number of Employees
Productivity per Employee
Global Presence (e.g., Number of Locations or Countries)
Supply Chain Efficiency
5. Industry-Specific Metrics
For Tech: Monthly Active Users (MAU), Daily Active Users (DAU), R&D Spend
For Retail: Same-store Sales, Average Order Value (AOV)
For Manufacturing: Production Volume, Unit Cost
For Healthcare: Patient Satisfaction, New Drugs/Products Approved
6. Branding and Reputation
Brand Value or Recognition
Media Sentiment
Customer Reviews and Ratings
Awards and Certifications
7. Growth and Strategy
Year-over-Year (YoY) Growth Rate
Strategic Initiatives (e.g., Mergers, Acquisitions, Expansions)
Innovation Index (e.g., Patents Filed)
8. Sustainability and Ethics
ESG (Environmental, Social, and Governance) Score
Carbon Footprint or Green Initiatives
Community Engagement
Diversity and Inclusion Metrics
9. Leadership and Organizational Health
Leadership Reputation (e.g., CEO/Board Ratings)
Employee Satisfaction (e.g., Glassdoor Ratings)
Organizational Structure (e.g., Hierarchical vs. Flat)
10. Competitive Positioning
Strengths, Weaknesses, Opportunities, Threats (SWOT)
Pricing Strategy (e.g., Premium vs. Competitive Pricing)
Barriers to Entry in the Industry
11. Technology and Innovation
Investment in R&D
Use of Emerging Technologies (e.g., AI, Blockchain)
Patents and Proprietary Technologies
12. Legal and Compliance
Legal Cases or Controversies
Adherence to Regulatory Standards
Licensing and Certifications"""

    prompt_text = f"""I have information about the company {companyname} - for which i  have information in {main_companyinfo},  against the competitors {competitors} - information about them is found in {competitor_info},  and I want a comparison based on specific metrics in markup language. If there is no data available for any metrics, please exclude that. Please generate a detailed table comparing the companies only, without additional textual explanations. Response table must be in markup language.

    Company names should be columns and features should be rows of the table.
Here is the list of metrics for comparison:
1. **Financial Performance**:
   - Revenue (in million $)
   - Net Income (in million $)
   - Profit Margin (%)
   - Market Capitalization (in billion $)

2. **Market and Customer Metrics**:
   - Market Share (%)
   - Customer Base (number of active customers)
   - Customer Retention Rate (%)

3. **Product and Service Offerings**:
   - Number of Products/Services Offered
   - Key Differentiators

4. **Operational Metrics**:
   - Number of Employees
   - Productivity per Employee (Revenue/Employee)
   - Global Presence (number of countries or locations)

5. **Sustainability and Ethics**:
   - ESG Score (out of 100)
   - Carbon Footprint (in metric tons)
   - Diversity and Inclusion Score



**Output Requirements**:
- Generate a table comparing the companies using the metrics above.
- Include all companies as rows and metrics as columns.
- Use markup language for creating the table .

Comparison should be based on:
 Revenue (in million $) 
 Net Income (in million $) 
 Domain competancies 
Products developed 
 Key Geographies 
 Number of Employees 
 Retention Rate (%) 
 Number of Products 
 Differentiators 
 Key Personnel with their designations
 Global Presence (countries)
 Key Differentiators 
 Key Risks facing 

Do not include any text or explanation outside this table. Only output the table. Also create the response in markup language
"""
    prompt_text1 = f"""I have information about the company {companyname} - for which i  have information in {main_companyinfo},  against the competitors {competitors} - information about them is found in {competitor_info},  and I want a comparison based on specific metrics . If there is no data available for any metrics, please exclude that. Please generate a detailed table comparing the companies only, without additional textual explanations. Response table must be in markup language.

        Company names should be columns and features should be rows of the table.
    Here is the list of metrics for comparison:{comparisonMetrics}
    **Output Requirements**:
- Generate a table comparing the companies using the metrics above.
- Include all companies as rows and metrics as columns.
- Use markup language for creating the table ."""
    
    prompt_text3 = f"""I have information about the company {companyname} - for which i  have information in {main_companyinfo},  against the competitors {competitors} - information about them is found in {competitor_info},  and I want neatly formatted  textual comparison based on specific metrics . If there is no data available for any metrics, please exclude that.

    Here is the list of metrics for comparison:{comparisonMetrics}. If no data is present for any of the metrics, exclude that particular metric.
    **Output Requirements**:


- Need only text based comparison, not table."""
    #draw comparison table
    prompt = f"Draw a detailed table  of comparison {companyname} with their competitors {competitors}. Generate the output in a table format using columns for each company. Do not include any textual explanation outside the table. Comparison metrics should have almost all details mentioned in {comparisonMetrics}. {companyname}'s details found in {main_companyinfo}, and competitor data found  in {competitor_info}. Make sure the comparison is done on all perspectives of input given, and no data is missed. But if data is not available for a perspective/feature, dont mention that perspective/feature. Response must be a TABLE, with borders on all cells. Also create the response in markup language"
    #prompt = prompt_text
    #prompt =f"For the company - {companyname}, we have to generate comparison table where data about the company is : {main_companyinfo} , competitor companies information can be found in {competitor_info}. Draw a neatly formatted, detailed comparison table.  Make sure you dont miss any of the information specified in the input data."

    response = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt, config={'temperature': 0.1})
    response1 = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt_text, config={'temperature': 0.1})
    response2 = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt_text1, config={'temperature': 0.1})
    response3 = client.models.generate_content(model='gemini-2.0-flash-exp',contents = prompt_text3, config={'temperature': 0.7})

    detailed_company_report_table = response.candidates.pop(0).content.parts.pop(0).text
    detailed_company_report_table1 = response1.candidates.pop(0).content.parts.pop(0).text
    detailed_company_report_table2 = response2.candidates.pop(0).content.parts.pop(0).text
    detailed_company_report_table3 = response3.candidates.pop(0).content.parts.pop(0).text

    #print(detailed_company_report_table)
    #print(detailed_company_report_table1)
    html_string = markdown2.markdown(detailed_company_report_table )
    html_string1 = markdown2.markdown(detailed_company_report_table1)
    html_string2 = markdown2.markdown(detailed_company_report_table2)
    html_string3 = markdown2.markdown(detailed_company_report_table3)

    # Define CSS for landscape layout
    css = CSS(string="""
        @page {
            size: A4 landscape; /* Set the page size and orientation */
            margin: 20mm;       /* Optional: Set custom margins */
        }
        th, td {
        border: 1px solid black; /* Adds solid borders to all cells */
        padding: 8px; /* Adds padding for better readability */
        text-align: center; /* Optional: Centers text in table cells */
        }
        th {
            background-color:rgb(242, 242, 242); /* Optional: Adds background color for headers */
        }
        """)

    # Convert the HTML to PDF with the specified CSS
    output_pdf = f"{companyname}\{companyname}_detailed_withcompetitorInfoTable.pdf"
    HTML(string=html_string).write_pdf(output_pdf, stylesheets=[css])

      # Convert the HTML to PDF with the specified CSS
    output_pdf = f"{companyname}\{companyname}_detailed_withcompetitorInfoTable1.pdf"
    HTML(string=html_string1).write_pdf(output_pdf, stylesheets=[css])

      # Convert the HTML to PDF with the specified CSS
    output_pdf = f"{companyname}\{companyname}_detailed_withcompetitorInfoTable2.pdf"
    HTML(string=html_string2).write_pdf(output_pdf, stylesheets=[css])

     # Convert the HTML to PDF with the specified CSS
    output_pdf = f"{companyname}\{companyname}_detailed_withcompetitorInfoTextual.pdf"
    HTML(string=html_string3).write_pdf(output_pdf)

    quadrantString 





client = qdrant_client.QdrantClient(
    url=qdrant_url,
    port=6333,
    api_key=qdrant_api_key,
)


#client = qdrant_client.QdrantClient(
#    url=qdrant_url,
#    port=6333,
#    api_key=qdrant_api_key,
#    timeout=30,
#)
#documents = SimpleDirectoryReader("astro/", recursive=True).load_data()
#documents = documents.append(excel_data)


vector_store = QdrantVectorStore(client=client, collection_name="CompanyAnalysis", enable_hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store)
splitter = SentenceSplitter(
    chunk_size=255,
    chunk_overlap=10,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
#print(quadrantString)
documents = [Document(text=quadrantString)]

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[splitter],
    
)
# Step 5: Create Chat Engine
chat_engine = index.as_chat_engine(chat_mode="context", system_prompt="""
    You are a helpful chatbot that provides accurate and detailed answers. 
    You understand and respond based on the uploaded documents in various formats.
    Answer the questions which is relevant only to the knowledge base provided. If any questions outside the context is asked, feel free to scold the user.
""")

# Step 6: Start the Chatbot
def chatbot():
    print("Chatbot started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        
        response = chat_engine.chat(user_input)
        print(f"Bot: {response}")

chatbot()





    

   
   