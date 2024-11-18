import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import os
import tqdm
import scipy.ndimage as nd
import utils


def dream(image, model, iterations, lr):
    tensor = torch.cuda.FloatTensor
    image = torch.autograd.Variable(tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = utils.clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep(image, model, iterations, lr, octaves_scale, num_octaves):
    image = utils.preprocess(image).unsqueeze(0).cpu().data.numpy()

    octaves = [image]

    for i in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octaves_scale, 1.0 / octaves_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        input_image = octave_base + detail
        i_have_a_dream = dream(image=input_image, model=model, iterations=iterations, lr=lr)
        detail = i_have_a_dream - octave_base

    return utils.deprocess(i_have_a_dream)


if __name__ == "__main__":
    path = r'C:\Users\ASUS\Desktop\深度学习\my_deep_dream\image'
    net = torchvision.models.vgg16(pretrained=True).cuda()
    features = list(net.features.children())
    model = torch.nn.Sequential(*features[:15])
    i = 0
    for file_name in os.listdir(path):
        if file_name.endswith('.jpg'):
            i += 1
            print(file_name)
            image = Image.open(os.path.join(path, file_name))

            image_np = np.array(image)
            image_np = image_np[:, :, :3]
            image = Image.fromarray(image_np)

            deep_image = deep(
                image=image,
                model=model,
                iterations=20,
                lr=0.01,
                octaves_scale=1.4,
                num_octaves=10
            )
            out_path = os.path.join(r'C:\Users\ASUS\Desktop\深度学习\my_deep_dream\deep_dream', 'ltd_{}.jpg'.format(i))
            deep_image = np.clip(deep_image, 0.0, 1.0)
            plt.imsave(out_path, deep_image)
            plt.imshow(deep_image)
            plt.show()
