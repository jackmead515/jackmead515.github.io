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
    
    # add styles to pre with style tag
    style = soup.new_tag('style', type='text/css')
    style.string = 'pre {overflow-y: auto;}'
    soup.head.append(style)
    
    # add styles to printed dataframes
    style = soup.new_tag('style', type='text/css')
    style.string = """
        table.dataframe > tbody > tr {
            white-space: nowrap !important;
        }
        table.dataframe > tbody > tr > th {
            white-space: nowrap !important;
        }
        table.dataframe > tbody > tr > td {
            white-space: nowrap !important;
            padding: 1em !important;
        }
    """
    soup.head.append(style)
    
    # find all table.dataframe
    for table in soup.find_all('table', class_='dataframe'):
        # get the parent div
        div = table.parent
        # edit the parent div
        div['style'] = 'overflow-y: auto;'

    # export
    with open(file_path, 'w') as file:
        file.write(str(soup))