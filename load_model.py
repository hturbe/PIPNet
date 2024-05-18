import torch

from pipnet.pipnet import PIPNet, get_network
from util.args import get_args
from util.data import get_dataloaders


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (
    trainloader,
    trainloader_pretraining,
    trainloader_normal,
    trainloader_normal_augment,
    projectloader,
    testloader,
    test_projectloader,
    classes,
) = get_dataloaders(args, device)


    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = (
        get_network(len(classes), args)
    )

    # Create a PIP-Net
    net = PIPNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    net = net.to(device=device)
    checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    # rename all key by replacing "module." with "_net"
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    net.load_state_dict(model_state_dict, strict=True)

    sample = trainloader.dataset[0]
    test = net(sample[0].unsqueeze(0).to(device))


if __name__ == "__main__":
    args = get_args()
    main(args)
