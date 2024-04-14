# Custom YOLOv7 with 2 heads

- Name: Nguyen Phuoc Nguyen
- Email: nguyenst279@gmail.com 

## Modify code
### models/yolo.py
Function: **parse_model**
Change layers from `d['backbone'] + d['head']` to `d['backbone']` to create model with only backbone

### custom_training.py
- Create `backbone` by load model and weight
- Create simple HeadA and HeadB

