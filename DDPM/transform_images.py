import os
from PIL import Image
import pandas as pd


IN_PATH = "data/data_for_fashion_clip/in"
OUT_PATH = "data/data_for_fashion_clip/out2"


for path in os.listdir(IN_PATH):
    if not path.endswith(".jpg"):
        continue
    # Add padding (left & right) to the image to fit 1750 pixels
    img = Image.open(os.path.join(IN_PATH, path))
    width, height = img.size
    new_color = img.getpixel((width // 2, 0))
    new_img = Image.new("RGB", (1750, 1750), new_color)
    new_img.paste(img, (int((1750 - width) / 2), int((1750 - height) / 2)))
    # Resize to 512x512
    new_img = new_img.resize((512, 512))
    new_img.save(os.path.join(OUT_PATH, path))

# Image to delete and remove from articles.csv
Image_to_delete = [
    '228257001',
    '481529006',
    '556350004',
    '583539003',
    '610234003',
    '612910001',
    '613919002',
    '617927001',
    '619235002',
    '622907001',
    '625131001',
    '632618006',
    '636193001',
    '636421001',
    '638927002',
    '640175001',
    '643588001',
    '647956001',
    '648821001',
    '648838001',
    '650502002',
    '661643002',
    '663355001',
    '663941001',
    '668579001',
    '675378002',
    '704805002'
]

Image_to_delete_int = list(map(int, Image_to_delete))

# Remove images from articles.csv
articles = pd.read_csv("data/data_for_fashion_clip/articles.csv")
articles = articles[~articles["article_id"].isin(Image_to_delete_int)]
articles.to_csv("data/data_for_fashion_clip/articles.csv", index=False)

# Remove images from out2
for path in Image_to_delete:
    os.remove(os.path.join(OUT_PATH, path + ".jpg"))
