# Overhead person detector (YOLO)

Single-class `person` model for the nadir camera on real flights. The Gazebo
survivors are red capsules, so the sim keeps using the HSV detector; this
model is selected with `NIDAR_DETECTOR=yolo`.

## Data

| Source | What it adds |
|---|---|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | Real drone footage, many altitudes/angles. `pedestrian` + `people` merged into `person`; frames without people kept as negatives. |
| C2A (Combination-to-Application) | Humans composited into disaster scenes (collapse, flood, fire), mostly top-down. |
| SARD ([Kaggle mirror](https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue), MIT) | SAR actors standing/sitting/lying in grass, quarry, forest. Export is 3x3 tiles stretched to 640x640; squashed back to 640x360. |
| HERIDAL ([Roboflow](https://universe.roboflow.com/work-950o2/heridal-lrbkc-1o7kq), CC BY 4.0, v1) | 4000x3000 Mediterranean wilderness SAR images at ~2 cm/px, tiled at native resolution (640 px, 20% overlap). |

SARD needs a Kaggle token (`~/.kaggle/access_token`), HERIDAL a Roboflow API
key. Use the full-resolution HERIDAL version above: the other Roboflow copies
are stretched to 640x640, which shrinks people to 2-3 px.

```bash
python3 prepare_dataset.py --visdrone ~/nidar/datasets/VisDrone \
    --c2a ~/nidar/datasets/C2A \
    --sard ~/nidar/datasets/SARD_raw/search-and-rescue \
    --heridal ~/nidar/datasets/HERIDAL_raw \
    --out ~/nidar/datasets/person_aerial_v2
```

## Train

```bash
python3 train_person_detector.py --data ~/nidar/datasets/person_aerial_v2/person_aerial.yaml \
    --model yolo26s.pt --batch 8 --epochs 60 --name yolo26s_person_aerial_v2
cp runs/yolo26s_person_aerial_v2/weights/best.pt ../models/person_aerial.pt
```

Flights are recorded on the Pi and processed afterwards on a GPU, so the
larger `yolo26s` is used. Batch 8 keeps the dense C2A frames inside 8 GB VRAM.
`yolo26n` remains the option if detection ever moves onboard the Pi 5.

## Results (YOLO26s, 60 epochs, `models/person_aerial.pt`)

Held-out test split, 640 px:

| Source | Images | mAP50 | Precision | Recall |
|---|---|---|---|---|
| SARD | 570 | 0.912 | 0.931 | 0.850 |
| HERIDAL (tiles) | 792 | 0.928 | 0.876 | 0.870 |
| C2A | 2043 | 0.842 | 0.860 | 0.787 |
| VisDrone | 1610 | 0.305 | 0.504 | 0.294 |
| **All** | 5015 | 0.711 | 0.831 | 0.642 |

VisDrone is dense oblique street crowds, not the mission profile; the SAR
sets (SARD, HERIDAL) are. A synthetic 15 m flight over a held-out HERIDAL
scene through `process_flight.py` found 19/20 people with a median geotag
error of 0.07 m (perfect telemetry; expect GPS error on top in real flights).

`models/person_aerial_n_prelabel.pt` is an earlier nano checkpoint trained on
VisDrone + C2A only, kept for pre-labelling footage in Label Studio.

## Deploy on the Raspberry Pi 5

The Pi 5 has no GPU, so the nano model runs on its CPU through NCNN. Export
on any machine (the output is portable) and copy it next to the .pt:

```bash
yolo export model=../models/person_aerial.pt format=ncnn imgsz=640
# -> ../models/person_aerial_ncnn_model/   (picked up automatically)
```

On the Pi: `pip install ultralytics ncnn`, then

```bash
NIDAR_DETECTOR=yolo python3 rescueswarm_mission.py
# optional: NIDAR_YOLO_WEIGHTS=/path/model NIDAR_YOLO_CONF=0.35
```

Frame rate is not critical here: at 8.5 m and 4 m/s a person stays in the
nadir camera's ~13 m along-track footprint for about 3 s, so even a few
frames per second gives several looks at each survivor. Run the Pi with
active cooling, or it throttles within minutes under sustained inference.

Quick check on images: `python3 ../scripts/person_detector.py <image-or-dir> [weights]`
writes `*_det.jpg` next to each input.
