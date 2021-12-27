
import albumentations as albu

def default_tfms(image_size):
    return [
        albu.RandomCrop(height=image_size, width=image_size, 
                        always_apply=True),
    ]

def aug_tfms(image_size):
    return [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.,
                              scale_limit=0.3,
                              rotate_limit=45,
                              p=.7,
                              border_mode=0),
        albu.PadIfNeeded(min_height=520, min_width=520,
                         always_apply=True, border_mode=0),
        albu.RandomCrop(height=image_size, width=image_size,
                        always_apply=True),
        albu.GaussNoise(p=.5),
        albu.Perspective(p=.2),
        albu.OneOf(
            [
                albu.CLAHE(p=.5),
                albu.RandomBrightnessContrast(p=.5),
                albu.RandomGamma(p=.2)
            ],
            p=0.4),
        albu.OneOf(
            [
                albu.Sharpen(p=.3),
                albu.Blur(blur_limit=5, p=.3),
                albu.MotionBlur(blur_limit=5, p=.3)
            ],
            p=0.5),
    ]
