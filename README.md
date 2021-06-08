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

## Downloading data
The partial data of chairs (03001627) has been uploaded.  
`data.tar.gz` can be downloaded from 
[this link](https://drive.google.com/file/d/1xf8V3aHtaTNdl6Gq8inBu15MpYSHNzWa/view?usp=sharing), 
and the corresponding file `sv2_chairs_train_partial.json` is provided 
[here](https://drive.google.com/file/d/1ZZ0JBGgCotW4YwBsaZpkhomlzpJDfukB/view?usp=sharing).  
Alternatively, run 
```bash
bash download_partial_data.sh
```
to get both the two files at once.

Data Layouts

<data_source_name>/
    .datasources.json
    SdfSamples/
        <dataset_name>/
            <class_name>/
                <instance_name>.npz
