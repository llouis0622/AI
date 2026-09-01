# 05. 객체 탐지

**만드는 것**: 탐지의 핵심 연산(IoU, NMS)을 직접 구현해 원리를 확인한 뒤, torchvision Faster R-CNN을 커스텀 데이터에 파인튜닝하고, 실무 표준인 YOLO 사용법까지 잇는다.

**선행 지식**: [IoU, NMS, 앵커, mAP](/handbook/07-computer-vision/07-object-detection-fundamentals), [탐지 아키텍처](/handbook/07-computer-vision/08-detection-architectures)

## 1. 핵심 연산 밑바닥 구현

탐지 모델이 무엇을 쓰든, 파이프라인의 앞뒤에는 반드시 IoU와 NMS가 있다. 이 둘을 못 짜면 탐지 결과를 다룰 수 없다.

```python
"""IoU와 NMS — 탐지의 두 기본 연산. 박스 형식: (x1, y1, x2, y2)"""
import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """boxes1 (N,4), boxes2 (M,4) → IoU 행렬 (N,M). 브로드캐스팅으로 루프 없이."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])   # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])   # (M,)
    # 교집합 박스: 왼쪽 위는 max, 오른쪽 아래는 min
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])                # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])                # (N,M,2)
    wh = (rb - lt).clamp(min=0)                     # 겹치지 않으면 0
    inter = wh[..., 0] * wh[..., 1]                 # (N,M)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5):
    """Non-Maximum Suppression: 점수 높은 박스부터 채택, 많이 겹치는 후보는 제거."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        best = order[0]
        keep.append(best.item())
        if order.numel() == 1:
            break
        ious = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_thresh]       # 임계값 이하만 생존
    return torch.tensor(keep)


# 검증: torchvision 공식 구현과 대조
if __name__ == "__main__":
    from torchvision.ops import nms as tv_nms, box_iou as tv_iou
    b = torch.rand(50, 2) * 100
    boxes = torch.cat([b, b + torch.rand(50, 2) * 50 + 5], dim=1)
    scores = torch.rand(50)
    assert torch.allclose(box_iou(boxes, boxes), tv_iou(boxes, boxes), atol=1e-5)
    assert nms(boxes, scores, 0.5).tolist() == tv_nms(boxes, scores, 0.5).tolist()
    print("torchvision 구현과 일치 ✓")
```

NMS가 필요한 이유: 탐지 모델은 한 물체 주변에 수십 개의 중복 박스를 출력한다. "가장 자신 있는 박스를 남기고, 그와 많이 겹치는(같은 물체일) 후보를 지운다"가 NMS의 전부다.

## 2. Faster R-CNN 파인튜닝

torchvision의 사전학습 탐지 모델은 헤드만 갈아끼우면 커스텀 클래스에 바로 파인튜닝된다. 데이터셋은 `(image, target)` 쌍을 반환해야 하고, target은 `boxes (N,4)`와 `labels (N,)`를 가진 딕셔너리다.

```python
"""Faster R-CNN 파인튜닝 — PennFudan 보행자 데이터셋 (2 클래스: 배경+사람)."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class PennFudan(Dataset):
    """https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip 를 ./data에 풀어둔다."""
    def __init__(self, root="./data/PennFudanPed"):
        self.root = Path(root)
        self.imgs = sorted((self.root / "PNGImages").iterdir())
        self.masks = sorted((self.root / "PedMasks").iterdir())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = read_image(str(self.imgs[i])) / 255.0
        mask = read_image(str(self.masks[i]))[0]        # 인스턴스 id 마스크
        ids = mask.unique()[1:]                          # 0은 배경
        boxes = []
        for obj_id in ids:                               # 인스턴스 마스크 → 바운딩 박스
            ys, xs = torch.where(mask == obj_id)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        target = {
            "boxes": tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32),
                                              format="XYXY", canvas_size=img.shape[-2:]),
            "labels": torch.ones(len(ids), dtype=torch.int64),  # 전부 '사람'(=1)
        }
        return img, target


def collate(batch):
    return tuple(zip(*batch))  # 탐지는 이미지 크기가 제각각 → 리스트로 묶는다


def main():
    ds = PennFudan()
    n_val = 30
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)

    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # 헤드 교체: COCO 91 클래스 → 우리 2 클래스(배경 + 사람)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.to(device)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-3, momentum=0.9, weight_decay=5e-4)

    for epoch in range(5):
        model.train()
        for imgs, targets in train_dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # 학습 모드의 탐지 모델은 손실 딕셔너리를 반환한다 (RPN 손실 + ROI 손실)
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"epoch {epoch+1}  loss {loss.item():.4f}")

    # 추론: 이미지만 넣으면 boxes/labels/scores를 반환한다
    model.eval()
    img, _ = ds[0]
    with torch.no_grad():
        pred = model([img.to(device)])[0]
    keep = pred["scores"] > 0.8
    print(f"검출 {keep.sum().item()}개:", pred["boxes"][keep].cpu().numpy().round(1))


if __name__ == "__main__":
    main()
```

## 3. 실무에서는: YOLO

프로토타이핑과 배포 속도가 중요하면 ultralytics YOLO가 사실상 표준이다.

```python
# pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n.pt")                    # 사전학습 모델 로드
results = model("street.jpg")                 # 추론 한 줄
results[0].show()                             # 박스 그려 보기

# 커스텀 데이터 파인튜닝: data.yaml에 클래스와 경로만 적으면 된다
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16)
metrics = model.val()                         # mAP50-95 등 자동 평가
model.export(format="onnx")                   # 배포용 내보내기
```

`data.yaml`은 이런 형식이다.

```yaml
path: ./dataset
train: images/train
val: images/val
names:
  0: person
  1: car
```

레이블은 이미지당 `.txt` 하나, 한 줄에 `class cx cy w h`(0~1 정규화)다. 이 포맷(YOLO 포맷)은 레이블링 도구 대부분이 지원한다.

## 확장 과제

1. **mAP 계산기 구현** — 예측 박스·점수와 GT로 AP@0.5를 계산하는 함수를 직접 짜고 `torchmetrics.detection.MeanAveragePrecision`과 대조하라. precision-recall 곡선을 그려 보면 mAP의 의미가 잡힌다.
2. **혼동 사례 분석** — 검출 실패(미검출/오검출)를 20장 모아 보라. 원인이 데이터(레이블 누락)인가 모델(작은 물체)인가? 이 습관이 [실패 분석](/handbook/14-practitioner-guide/04-training-debug-checklist)의 기본기다.
3. **Soft-NMS** — 겹치는 박스를 제거하는 대신 점수를 감쇠시키는 Soft-NMS를 구현해, 겹친 물체가 많은 장면에서 비교하라.

## 다음

CNN을 벗어나 Transformer로 이미지를 본다 → [06. ViT 밑바닥 구현](/practice/06-vit-from-scratch)
