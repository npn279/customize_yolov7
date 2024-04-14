# Custom YOLOv7 with 2 heads

- Name: Nguyen Phuoc Nguyen
- Email: nguyenst279@gmail.com 

## Modify code
### models/yolo.py
Function: **parse_model**
Change layers from `d['backbone'] + d['head']` to `d['backbone']` to create model with only backbone

### custom_training.py
1. Create `backbone` by load model and weight
2. Create simple **HeadA** and **HeadB** in the **MultitaskModel** with a Pooling and a Dense
3. Create dataset, dataloader, training, inference scripts 

- `instances_val2017.json` downloaded from https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
- Train test size: **90% train : 10% val**
- Training: **num_epochs=3, batch_size=64, optimizer=Adam, learning_rate=0.01**

4. Train model on 2 tasks, freeze `backbone` and one `head` when training on the other head
5. Save model

