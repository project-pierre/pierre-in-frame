import gdown
output = 'dataset.zip'
url = 'https://drive.google.com/uc?id=1mc7ZsI5HEeLwbDRJ3f8JHUWl2tjlljKh'
gdown.download(url, output, quiet=False)
