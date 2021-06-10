import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsons_root', type=str, default="./shapenet/splits/", help="The path to the json files that define data splitting")
    parser.add_argument('--data_root', type=str, default="./shapenet/data/", help="The path to the shapenet data")
    parser.add_argument('--output_root', type=str, default="./shapenet/output/", help="The path to output the rendered images")
    parser.add_argument('--views', type=int, default=50, help="How many views are attemped to be generated")
    parser.add_argument('--maximum_try', type=int, default=200, help="To avoid infinite loop")
    args = parser.parse_args()

    for json_file in sorted(os.listdir(args.jsons_root)):

        with open(os.path.join(args.jsons_root, json_file), "r") as f:
            data_info = json.load(f)
        data_contained = data_info["ShapeNetV2"]

        for key, value in data_contained.items():
            data_id = key
            data_list = value

        for idx, data_name in enumerate(data_list):
            try:
                data_path = os.path.join(args.data_root, data_id, data_name, "models", "model_normalized")

                obj_image_easy_dir = os.path.join(args.output_root, data_id, data_name, "easy", "image")
                obj_albedo_easy_dir = os.path.join(args.output_root, data_id, data_name, "easy", "albedo")
                obj_image_hard_dir = os.path.join(args.output_root, data_id, data_name, "hard", "image")
                obj_albedo_hard_dir = os.path.join(args.output_root, data_id, data_name, "hard", "albedo")

                os.makedirs(obj_image_easy_dir) 
                os.makedirs(obj_albedo_easy_dir) 
                os.makedirs(obj_image_hard_dir) 
                os.makedirs(obj_albedo_hard_dir) 

                receive = os.system(f"blender --background --python render.py -- --obj {data_path} --views {args.views} --maximum_try {args.maximum_try} --obj_image_easy_dir {obj_image_easy_dir} --obj_albedo_easy_dir {obj_albedo_easy_dir} --obj_image_hard_dir {obj_image_hard_dir} --obj_albedo_hard_dir {obj_albedo_hard_dir}")
                
                if receive == 1:
                    os.system(f"echo {data_path} >> {os.path.join(args.output_root, 'error.txt')}")
                    os.system(f"echo infinite loop >> {os.path.join(args.output_root, 'error.txt')}")

            except Exception as e:
                os.system(f"echo {data_path} >> {os.path.join(args.output_root, 'error.txt')}")
                os.system(f"echo {e} >> {os.path.join(args.output_root, 'error.txt')}")
            break

        break