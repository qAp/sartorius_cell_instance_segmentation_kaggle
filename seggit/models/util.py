
import segmentation_models_pytorch as smp


def create_segmentation_model(data_config, args=None):

    arch = getattr(smp, args.arch)
    encoder_name = args.encoder_name
    in_channels = data_config['input_dims'][2]
    classes = data_config['output_dims'][2]

    if classes == 3:
        activation = 'softmax'
    else:
        activation = 'sigmoid'

    model = arch(encoder_name=encoder_name,
                 in_channels=in_channels,
                 classes=classes,
                 activation=activation)

    return model


    
