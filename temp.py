import json

with open('instances_val2017.json', 'r') as f:
    annotations = json.load(f)

cats = [x['category_id'] for x in annotations['annotations']]
classes = sorted(list(set(cats)))
print(len(classes))

