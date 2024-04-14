# Custom YOLOv7 with 2 heads

- Name: Nguyen Phuoc Nguyen
- Email: nguyenst279@gmail.com 

## Modify code
### models/yolo.py
Function: **parse_model**
Change layers from `d['backbone'] + d['head']` to `d['backbone']` to create model with only backbone

### custom_training.py
1. Create `backbone` by load model and weight
2. Create simple **HeadA** and **HeadB** in the **MultitaskModel**
3. Create dataset, dataloader, training, inference scripts 


