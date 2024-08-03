import sys
import bs4

if __name__ == '__main__':
    
    file_path = sys.argv[1]
    
    # replace ipynb extension with html
    file_path = file_path.replace('.ipynb', '.html')
    
    with open(file_path, 'r') as file:
        html = file.read()
        
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # get body content
    body = soup.find('body')
    
    # add styles to body
    body['style'] = 'max-width: 1000px; margin: 0 auto; padding-bottom: 50vh;'
    
    # export
    with open(file_path, 'w') as file:
        file.write(str(soup))