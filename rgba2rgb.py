import os
import glob
from PIL import Image

classes ={
    'chairs':'03001627'
}


if __name__ == '__main__':

    data_source = './data'
    class_name = classes['chairs']
    level = 'easy'

    import argparse

    arg_parser = argparse.ArgumentParser(description="Convert RGBA image to RGB")
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default = './data',
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--class_name",
        "-c",
        dest = "class_name",
        default = 'chairs',
        help="catogry name",
    )
    arg_parser.add_argument(
        "--level",
        dest="level",
        default='easy',
        help="easy/hard",
    )

    args = arg_parser.parse_args()

    class_id = classes[args.class_name]
    level = args.level
    data_source = args.data_source




    images_path = os.path.join(data_source, class_id)
    for instance_dir in glob.glob(os.path.join(images_path, '*')):
        rgb_dir = os.path.join(instance_dir,level, 'image_rgb')
        rgba_dir =os.path.join(instance_dir, level, 'image')
        os.makedirs(rgb_dir, exist_ok=True)
        print(instance_dir)
        for img_path in glob.glob(os.path.join(rgba_dir,'*.png')):
            img_name = img_path.split("/")[-1]
            rgba_image = Image.open(img_path)
            rgba_image.load()
            background = Image.new("RGB", rgba_image.size, (255, 255, 255))
            background.paste(rgba_image, mask = rgba_image.split()[3])
            background.save(os.path.join(rgb_dir, img_name), "png", quality=100)
            

            
        
