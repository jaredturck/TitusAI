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
    
    for i in range(10):
        yield None

with open('training_data.csv', mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    counter = 0
    gen = get_dialog()
    start = time.time()
    src = 'start'
    target = 'start'

    while src and target:
        src = next(gen)
        target = next(gen)
        writer.writerow([src, target])
        counter += 1

        if time.time() - start > 2:
            start = time.time()
            print(f'[+] Processed {counter} pairs')
