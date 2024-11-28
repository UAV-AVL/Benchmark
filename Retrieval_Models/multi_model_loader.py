
from Retrieval_Models.CAMP.get_CAMP import get_CAMP_model
from torchvision import transforms

###################################################################################################################
def get_transforms_new():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transforms

def get_Model(model_name):

    if model_name == 'CAMP':
        model = get_CAMP_model()
        val_transforms = get_transforms_new()
    return model, val_transforms








