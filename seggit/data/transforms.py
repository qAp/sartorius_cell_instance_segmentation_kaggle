
import albumentations as albu

def default_tfms(image_size):
    return [
        albu.RandomCrop(height=image_size, width=image_size, 
                        always_apply=True),
    ]

def aug_tfms(image_size):
    return [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.2,
                              scale_limit=0.3,
                              rotate_limit=180,
                              p=1.,
                              border_mode=0),
        albu.PadIfNeeded(min_height=520,
                         min_width=520,
                         always_apply=True,
                         border_mode=0),
        albu.RandomCrop(height=image_size,
                        width=image_size,
                        always_apply=True),
        albu.GaussNoise(p=1),
        albu.Perspective(p=1),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1)
            ],
            p=0.9),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=5, p=1),
                albu.MotionBlur(blur_limit=5, p=1)
            ],
            p=0.9),
        albu.HueSaturationValue(p=0.9)
    ]
