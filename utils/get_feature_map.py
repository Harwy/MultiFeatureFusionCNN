import os
import torchvision.utils as vutil

INSTANCE_FOLDER = "E:/FeatureAgeNet/analyse/feature_map"

def hook_func(module, input, output):
    """
    Hook function of register_forward_hook

    Parameters:
    -----------
    module: module of neural network
    input: input of module
    output: output of module
    """
    image_name = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    vutil.save_image(data, image_name, pad_value=0.5)

def get_image_name_for_hook(module):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(
            INSTANCE_FOLDER, '%s_%d.png' % (base_name, index))
    return image_name

if __name__ == "__main__":
    """
    how to use
    """
    import torch
    model = ""
    model.eval()

    modules_for_plot = (torch.nn.ReLU, torch.nn.Conv2d,
                            torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)
    for name, module in model.named_modules():
        if isinstance(module, modules_for_plot):
            module.register_forward_hook(hook_func)

    y =model(x)
    torch.sum(torch.sigmoid(y) > 0.5, dim=1)