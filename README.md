# Template-is-all-you-need
Template is all you need: 2D to 3D reconstruction with template learned by contrastive learning

## Prepare
1. Please prepare `config.json`, which should include:
```json
{"pascal_root":"path-to-save-PASCAL3D+_release1.1.zip",
 "pascal_folder":"path-of-PASCAL3D+_release1.1-folder"}
```
2. Utilize `initialize.py` to prepare everyting required.
```script
python3 initialize.py
```