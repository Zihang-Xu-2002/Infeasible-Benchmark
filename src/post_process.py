import re

def extract_num(response):

    pattern = re.compile(r"(?i)probability\s*[^0-9]*(\d+\.\d+)")

    # Search for the pattern in the text
    match = pattern.search(response)

    # Convert the captured value to float if a match is found, otherwise None
    prob = float(match.group(1)) if match else 'None'
    
    return prob