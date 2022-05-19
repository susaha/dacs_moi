import torch

'''
CODE EXPLANATION:
assumping the crop dim 500,500, i.e. H =500, W = 500

pred: {tensor: 500, 500} - pred is source GT semanitc label
classes: {tensor: (7,)}  - are the class label trainIds (0,1,...,15) present in the GT label (for source image)

for torch torch.broadcast_tensors, we need to align the tensor shape following pytroch braodcasting semantics, so apply unsqueeze for that
pred.unsqueeze(0): {tensor: 1, 500, 500}
classes.unsqueeze(1).unsqueeze(2): {tensor: 7, 1, 1}

after alinging the dimension now we can broadcast both pred and classes to reshape these two tensor in a common shape of 7,500,500

broad cast basically help filling up the empty places when we want to represent on tensor to the shape of another tensor
or when we want to present two different shape tensors in a common shape (like in this case)

please note, broadcast_tensors returns two new tensor t1_new and t2_new of the original tensors t1_ori, t2_ori
t1_new, t2_new = torch.broadcast_tensors(t1_ori, t2_ori)
Note, the new tensors t1_new and t2_new have the same values just the empty places are filled up
For example, before and after broascasting pred.unique() (or classes.unique()) return the same result
so the values in the tensor remains the same

N = pred.eq(classes).sum(0)
this above line basically says that I want to mask out only the classids present in the classes list
from the pred tensor.
pred.eq(classes) -> returns a boolen tensor of shape 7,500,500
but we need a 500x500 (height x widht) mask to generate our augmented images
so we take a sum along the first dim 
'''
# this create a label mask as per the class ids present in classes
def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N