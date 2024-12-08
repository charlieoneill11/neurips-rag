from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
import re

def type_slowly(page, selector, text):
    """Type text character by character with small delays"""
    print(f"Typing into {selector}...")
    for char in text:
        page.type(selector, char)
        time.sleep(0.1)

def login_to_neurips(page, username, password, url):
    """Handle the NeurIPS login process with deliberate delays"""
    try:
        print("Navigating to main page...")
        page.goto(url)
        time.sleep(2)
        
        print("Looking for login link...")
        login_link = page.locator('a[href^="/login"]')
        if login_link.is_visible():
            print("Found login link, clicking...")
            login_link.click()
        else:
            print("Login link not found!")
            return False
            
        time.sleep(3)
        
        print("Starting to fill login form...")
        print(f"Typing username: {username}")
        type_slowly(page, '#id_username', username)
        time.sleep(1)
        
        print(f"Typing password: {'*' * len(password)}")
        type_slowly(page, '#id_password', password)
        time.sleep(1)
        
        print("Finding login button...")
        submit_button = page.locator('button.btn.btn-primary.float-end[type="submit"][value="Log In"]')
        
        if submit_button.is_visible():
            print("Found login button, clicking...")
            time.sleep(1)
            submit_button.click()
            print("Clicked login button")
        else:
            print("Login button not found!")
            return False
        
        print("Waiting for navigation after login...")
        page.wait_for_load_state('networkidle')
        time.sleep(5)
        
        try:
            page.wait_for_selector(".track-schedule-card", timeout=10000)
            print("Login successful!")
            return True
        except Exception as e:
            print(f"Login check failed: {e}")
            return False
            
    except Exception as e:
        print(f"Error during login process: {e}")
        return False

def extract_session_info(soup):
    """Extract session metadata from the page"""
    session_info = {}
    
    # Extract session title
    title_elem = soup.find('h2', class_='card-title main-title text-center')
    if title_elem:
        session_info['session_title'] = title_elem.text.strip()
    
    # Extract location
    location_elem = soup.find('h5', class_='text-center text-muted')
    if location_elem:
        session_info['location'] = location_elem.text.strip()
    
    # Extract time information
    time_div = soup.find('div', string=re.compile(r'.*\d+:\d+ [pa]\.m\. PST.*'))
    if time_div:
        time_text = time_div.text.strip()
        # Extract date and time using regex
        match = re.search(r'((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d+\s+\w+)\s+(\d+:\d+\s+[pa]\.m\.\s+PST)\s+—\s+(\d+:\d+\s+[pa]\.m\.\s+PST)', time_text)
        if match:
            session_info['date'] = match.group(1)
            session_info['start_time'] = match.group(2)
            session_info['end_time'] = match.group(3)
    
    return session_info

def scrape_session(page, url):
    """Scrape a single session's posters and metadata"""
    try:
        print(f"\nScraping session: {url}")
        page.goto(url)
        page.wait_for_selector(".track-schedule-card", timeout=30000)
        time.sleep(2)  # Wait for dynamic content
        
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Get session information
        session_info = extract_session_info(soup)
        
        # Get posters
        posters = []
        poster_cards = soup.find_all('div', class_='track-schedule-card')
        
        print(f"Found {len(poster_cards)} posters")
        
        for card in poster_cards:
            try:
                poster_data = {}
                
                # Extract poster number
                poster_number_div = card.find('div', class_='small', title='Poster Position')
                if poster_number_div:
                    poster_number = poster_number_div.text.strip('#')
                    try:
                        poster_data['poster_number'] = int(poster_number)
                    except ValueError:
                        poster_data['poster_number'] = poster_number
                
                # Extract title and URL
                title_link = card.find('h5').find('a')
                if title_link:
                    poster_data['title'] = title_link.text.strip()
                    relative_url = title_link['href']
                    poster_data['url'] = "https://neurips.cc" + relative_url
                
                # Extract authors
                authors_p = card.find('p', class_='text-muted')
                if authors_p:
                    authors = [author.strip() for author in authors_p.text.split('·')]
                    poster_data['authors'] = authors
                
                # Extract abstract
                abstract_div = card.find('div', class_='abstract')
                if abstract_div:
                    abstract_paragraphs = abstract_div.find_all('p')
                    abstract_text = ' '.join(p.text.strip() for p in abstract_paragraphs if p.text.strip())
                    poster_data['abstract'] = abstract_text
                
                posters.append(poster_data)
                
            except Exception as e:
                print(f"Error processing poster card: {e}")
                continue
        
        return {
            "session_info": session_info,
            "posters": posters
        }
        
    except Exception as e:
        print(f"Error scraping session {url}: {e}")
        return None

def scrape_all_sessions(urls):
    """Scrape all poster sessions"""
    with sync_playwright() as p:
        chrome_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-web-security",
            "--disable-infobars",
        ]
        
        modern_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
        context = p.chromium.launch_persistent_context(
            user_data_dir="./chrome-data",
            headless=False,
            channel="chrome",
            args=chrome_args,
            viewport={"width": 1280, "height": 720},
            user_agent=modern_user_agent
        )
        
        page = context.new_page()

        config = yaml.safe_load(Path("config.yaml").read_text())
        username = config['username']
        password = config['password']
        
        # Login first
        login_success = login_to_neurips(
            page, 
            username,
            password,
            urls[0]
        )
        
        if not login_success:
            print("Login failed, cannot proceed with scraping")
            return None
        
        # Scrape all sessions
        sessions_data = {}
        
        for url in urls:
            session_data = scrape_session(page, url)
            if session_data:
                session_id = url.split('/')[-1]
                sessions_data[session_id] = session_data
        
        context.close()
        return sessions_data

def main():
    urls = [
        "https://neurips.cc/virtual/2024/session/108363",
        "https://neurips.cc/virtual/2024/session/108364",
        "https://neurips.cc/virtual/2024/session/108361",
        "https://neurips.cc/virtual/2024/session/108362",
        "https://neurips.cc/virtual/2024/session/108365",
        "https://neurips.cc/virtual/2024/session/108366",
        "https://neurips.cc/virtual/2024/session/108367",
        "https://neurips.cc/virtual/2024/session/108368",
        "https://neurips.cc/virtual/2024/session/108369",
        "https://neurips.cc/virtual/2024/session/108370",
        "https://neurips.cc/virtual/2024/session/108371",
        "https://neurips.cc/virtual/2024/session/108372"
    ]
    
    print("Starting scraper...")
    sessions_data = scrape_all_sessions(urls)
    
    if sessions_data:
        output_file = 'neurips_sessions.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sessions_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved all session data to {output_file}")
        
        # Print summary
        print("\nScraping Summary:")
        for session_id, data in sessions_data.items():
            print(f"\nSession {session_id}:")
            print(f"Title: {data['session_info'].get('session_title', 'N/A')}")
            print(f"Location: {data['session_info'].get('location', 'N/A')}")
            print(f"Posters: {len(data['posters'])}")
    else:
        print("Failed to scrape sessions")

if __name__ == "__main__":
    main()