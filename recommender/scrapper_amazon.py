import json

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent':
        'Mozilla/5.0 (X11; Linux x86_64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/42.0.2311.90 Safari/537.36'
}

URL = 'https://www.amazon.com/dp/'

# DEBUG LOCAL
# http://localhost:8000/amazon-detail/1196


def get_data(asin):
    page = requests.get(URL + asin, headers=HEADERS).text
    soup = BeautifulSoup(page, "lxml")

    try:
        title = soup.find("h1", {"id": "title"}).findChildren()[0].text
    except:
        title = None

    if soup.find("img", {"id": "landingImage"}) is not None:
        image_id = "landingImage"
    elif soup.find("img", {"id": "imgBlkFront"}) is not None:
        image_id = "imgBlkFront"
    elif soup.find("img", {"id": "ebooksImgBlkFront"}) is not None:
        image_id = "ebooksImgBlkFront"
    else:
        image_id = None

    if image_id is not None:
        image = soup.find("img", {"id": image_id}).get("data-a-dynamic-image")
        image = json.loads(image)
        image = list(image.keys())[0]
    else:
        image = None

    return {
        'asin': asin,
        'image': image,
        'title': title
    }

