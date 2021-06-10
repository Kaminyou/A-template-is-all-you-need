# Template-is-all-you-need
Template is all you need: 2D to 3D reconstruction with template learned by contrastive learning

## Prepare
1. Please prepare `secret.json`, which should include:
```json
{"pascal_root":"path-to-save-PASCAL3D+_release1.1.zip",
 "pascal_folder":"path-of-PASCAL3D+_release1.1-folder",
 "shapenet_url":"download-link-of-ShapeNetCore.v2.zip",
 "shapenet_root":"path-to-save-ShapeNetCore.v2.zip",
 "shapenet_folder":"path-of-ShapeNetCore.v2-folder"}
```
2. Utilize `initialize.py` to prepare everyting required.
```script
python3 initialize.py
```

## Render 2D images
The code is heavily based on [link](https://github.com/Xharlie/ShapenetRender_more_variation). However, only `images` and `albedo` defined there will be generated. You **should build the environment accurately first to avoid any erroneous 2D image rendering**!
1. Download [blender](https://download.blender.org/release/Blender2.79/) v2.79. You **must use v2.79** because some functions are deprecated in the latest version.
2. Test if the command `blender` works or not.
```script
blender
```
3. Install OpenCV in the blender python env. Please (cd) to [blender]/2.79/python/bin
```script
./python3.5m -m ensurepip
./python3.5m -m pip install --upgrade pip
./python3.5m -m pip install opencv-python
./python3.5m -m pip install opencv-contrib-python
```
4. Try to `import numpy` in this python env. If there is any error, re-install numpy by **deleting the numpy folder in the following path `[blender]/2.79/python/lib/site-packages/`**. Then,
```script
./python3.5m -m pip install numpy
```
5. Start generate the 2D images by the following command. Please notice that the `jsons_root` denotes `./examples/splits/`, `data_root` denotes `ShapeNetCore.v2/`, and `output_root` is where you want to save the 2D images.
```script
python batch.py --jsons_root $jsons_root --data_root $data_root --output_root $output_root
```

## Downloading data
The training data of chairs `03001627_train.tar.gz` can be downloaded from 
[this link](https://drive.google.com/file/d/17j9uOb3cVXm4sqHcRcgkPBFdCmsYAv3J/view?usp=sharing). 
There will be a folder named `03001627` when the file is extracted.  

To download the file and merge all the extracted data to your dataset folder, run  
```bash
bash download_chairs_train.sh [data_source_name]  
```  
where `[data_source_name]` should be the path to yor dataset folder. 
To be more explicit, it should be the parent folder of `SdfSamples`. 
Therefore, before running this script, please make sure that all the other data has been prepared 
and the folder structure is the same as the one specified in [Data Layouts](https://github.com/Kaminyou/Template-is-all-you-need#data-layouts).     

For example, run  
```bash
bash download_chairs_train.sh data/  
```  

###### Data Layouts
```script
<data_source_name>/
    .datasources.json
    SdfSamples/
        <dataset_name>/
            <class_name>/
                <instance_name>/
                      <instance_name>.npz
                      <image>/
                          XXXXXXXX.png
                          XXXXXXXX.png
                          .
                          .
                          .
                              
```

## Encoders

Encoders that apply different contrastive learning frameworks can be found in 
the folder `encoders`.  
For more details please refer to [ENCODER.md](encoders/ENCODER.md).  

## Training

```python
python train_deep_implicit_templates.py -e examples/cars_dit --debug --batch_split 2 -d ./data
```

## Acknowledgements
This code repo is heavily based on [Deep Implicit Template](https://github.com/ZhengZerong/DeepImplicitTemplates/tree/db65db3c22e0f5111236e48deab7cffb38bd60c3). We thank the authors for their great job!
