from torchvision import transforms
import PIL


def load_img_transforms():

    """
    Funcion que carga las transformaciones

    :return:
    """
    train_data_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.485,
                             std=0.225),
    ])

    val_data_transform = train_data_transform
    test_data_transform = train_data_transform

    transforms_dict = {
        "train": train_data_transform,
        "val": val_data_transform,
        "test": test_data_transform
    }

    return transforms_dict


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor