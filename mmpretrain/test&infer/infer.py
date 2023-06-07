from mmpretrain import ImageClassificationInferencer


test_image = '/home/xiaoqiang/mlearning/dataset/demo/fruit.jpeg'
model = '/home/xiaoqiang/mlearning/mmlab2/mmpretrain/fruit_work/fruit_resnet50_256_mixup_label.py'
checkpoint = '/home/xiaoqiang/mlearning/mmlab2/mmpretrain/work_dirs/fruit_resnet50_256_mixup_label/best_accuracy_top1_epoch_234.pth'
inferencer = ImageClassificationInferencer(model=model, pretrained=checkpoint)
result = inferencer(test_image)[0]
print(result['pred_class'])
