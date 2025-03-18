# 🗂️ CALVIN dataset

Downloading and preprocessing of CALVIN dataset

## Download

To download the debug and main dataset run from this directory

```
bash ./scripts/download.sh
```

# Preprocessing

**Note:** Activate play_segmentation environment


**Compute Requirements:**
- ~ 12 hrs, 1x 40GB A100
This code will:

- Create the dataset splits (10%,25%,50%).
- Embed the observations and instructions via CLIP (for UnLOC method).



```
python convert.py --config ./configs/convert.yaml
```
