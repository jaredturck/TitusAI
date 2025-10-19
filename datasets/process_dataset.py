import torch, os, csv, re, time
from bs4 import BeautifulSoup

html_files = [file for file in os.listdir('/home/jared/Documents/Dropdown Documents/TitusAI/datasets') if file.endswith('.html')]

def get_dialog():
    for html_file in html_files:
        with open(html_file, "r", encoding="utf-8") as file:
            html = file.read()

            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup.find_all('blockquote'):
                tag_content = tag.get_text(separator=' ', strip=True).replace('\n', ' ')
                tag_content = re.sub(r'[^a-zA-Z0-9 \'!"£$%^&*()_+-=\\[\]{};:\,./<>?@#~`|]+', ' ', tag_content)
                yield tag_content

with open('training_data.txt', mode='w', newline='', encoding='utf-8') as output_file:

    counter = 0
    start = time.time()
    src = 'start'
    target = 'start'

    for row in get_dialog():
        output_file.write(row + '\n')
        counter += 1

        if time.time() - start > 2:
            start = time.time()
            print(f'[+] Processed {counter} rows')
