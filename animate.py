import os
import time
from tqdm import tqdm

import imageio
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


images = []

for size in range(0, 6):
    image_names = os.listdir(f"previews/previews_{size}")
    image_names.sort(key=lambda x: int(x.split(".")[0]))

    for image_name in tqdm(image_names):
        image = Image.open(os.path.join(f"previews/previews_{size}", image_name))

        I1 = ImageDraw.Draw(image)
        I1.text(
            (25, 25),
            f'{size}_{image_name.split(".")[0]}',
            font=ImageFont.truetype("arial.ttf", 50),
            fill=(0, 0, 0)
        )

        images.append(image)
        if image_name == '40.png':
            for i in range(9):
                images.append(image)

start_time = time.time()
imageio.mimsave("training.gif", images)
print(f"Save time: {round(time.time() - start_time, 2)}s")
