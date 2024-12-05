from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import yaml
import time
from pathlib import Path

def read_config(config_path):
    """Read configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def type_slowly(page, selector, text):
    """Type text character by character with small delays"""
    print(f"Typing into {selector}...")
    for char in text:
        page.type(selector, char)
        time.sleep(0.1)  # 100ms delay between each character

def login_to_neurips(page, username, password, url):
    """Handle the NeurIPS login process with deliberate delays"""
    try:
        print("Navigating to main page...")
        page.goto(url)
        time.sleep(2)  # Wait for page to settle
        
        print("Looking for login link...")
        login_link = page.locator('a[href^="/login"]')
        if login_link.is_visible():
            print("Found login link, clicking...")
            login_link.click()
        else:
            print("Login link not found!")
            return False
            
        time.sleep(3)  # Wait for login page to load
        
        print("Starting to fill login form...")
        print(f"Typing username: {username}")
        type_slowly(page, '#id_username', username)
        time.sleep(1)  # Short pause between fields
        
        print(f"Typing password: {'*' * len(password)}")
        type_slowly(page, '#id_password', password)
        time.sleep(1)  # Pause before clicking submit
        
        print("Finding login button...")
        # More specific selector for the login button
        submit_button = page.locator('button.btn.btn-primary.float-end[type="submit"][value="Log In"]')
        
        if submit_button.is_visible():
            print("Found login button, clicking...")
            time.sleep(1)  # Short pause before clicking
            submit_button.click()
            print("Clicked login button")
        else:
            print("Login button not found!")
            print("Current page content:", page.content())  # Debug info
            return False
        
        print("Waiting for navigation after login...")
        page.wait_for_load_state('networkidle')
        time.sleep(5)  # Give plenty of time for the login to process
        
        # Check if login was successful by looking for the poster cards
        print("Checking if login was successful...")
        try:
            page.wait_for_selector(".track-schedule-card", timeout=10000)
            print("Login successful! Found poster cards.")
            return True
        except Exception as e:
            print(f"Login check failed: {e}")
            print("Couldn't find poster cards after login")
            return False
            
    except Exception as e:
        print(f"Error during login process: {e}")
        return False

def scrape_neurips_posters(url, user_data_dir):
    """
    Scrape poster information from NeurIPS using Playwright with login
    """
    with sync_playwright() as p:
        # Chrome arguments to disable security warnings and maintain session
        chrome_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-web-security",
            "--disable-infobars",
        ]
        
        # Modern Chrome user agent
        modern_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            channel="chrome",
            args=chrome_args,
            viewport={"width": 1280, "height": 720},
            user_agent=modern_user_agent
        )
        
        page = context.new_page()
        
        try:
            # Handle login first
            login_success = login_to_neurips(
                page, 
                "charles.oneill@anu.edu.au", 
                "Reason211!", 
                url
            )
            
            if not login_success:
                print("Login failed, cannot proceed with scraping")
                return []
            
            # Get the page content after JavaScript execution
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            posters = []
            poster_cards = soup.find_all('div', class_='track-schedule-card')
            
            print(f"Found {len(poster_cards)} poster cards")
            
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
                    print(f"Error processing a poster card: {e}")
                    continue
            
        except Exception as e:
            print(f"Error during scraping: {e}")
            return []
        
        finally:
            context.close()
            
        return posters

def save_to_json(posters, output_file='neurips_posters.json'):
    """Save the scraped poster data to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(posters, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved data to {output_file}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")

def main():
    # Read configuration
    config = read_config('config.yaml')
    user_data_dir = config['user_data_dir']
    
    url = "https://neurips.cc/virtual/2024/session/108363"
    
    print("Starting scraper...")
    posters = scrape_neurips_posters(url, user_data_dir)
    
    if posters:
        print(f"Successfully scraped {len(posters)} posters")
        save_to_json(posters)
        
        print("\nSample poster data:")
        print(json.dumps(posters[0], indent=2))
    else:
        print("No posters found. Login may have failed.")

if __name__ == "__main__":
    main()