from PIL import Image
import os
import random


def prerprocess_crop_resize_single(imgfile, size):
    img = Image.open(imgfile)
    x, y = img.size
    crop_size = min(x, y)
    if x == crop_size:
        offset = random.randint(0, y-crop_size)
        box = (0, offset, crop_size, crop_size+offset)
        newimg = img.crop(box)
    else:
        offset = random.randint(0, x-crop_size)
        box = (offset, 0, crop_size+offset, crop_size)
        newimg = img.crop(box)
    newimg = newimg.resize((size, size))
    return newimg


def prerprocess_crop_resize_dir(imgdir, dstdir, size):
    for filename in os.listdir(imgdir):
        if filename == '.DS_Store':
            continue
        srcpath = os.path.join(imgdir, filename)
        dstpath = os.path.join(dstdir, filename)
        try:
            new_img = prerprocess_crop_resize_single(srcpath, size)
            if new_img.mode == "RGB" or new_img.mode == "1":
                new_img.save(dstpath, "JPEG")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    prerprocess_crop_resize_dir("data/waifus/images", "data/waifus/processed", 250)