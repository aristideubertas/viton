import math
from PIL import Image

def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    """
    Resize an image to fit within the specified dimensions while maintaining aspect ratio,
    then pad to fill the target dimensions.
    
    Args:
        im: PIL Image to resize and pad
        new_width: Target width
        new_height: Target height
        pad_color: Color to use for padding (RGB tuple)
        mode: PIL resize filter mode
        
    Returns:
        Tuple of (padded image, padding width, padding height)
    """
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    """
    Remove padding from an image and resize back to original dimensions.
    
    Args:
        padded_im: Padded PIL Image
        pad_w: Padding width
        pad_h: Padding height
        original_width: Original image width
        original_height: Original image height
        
    Returns:
        Unpadded and resized PIL Image
    """
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))

    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

def resize_image(img, target_size=768):
    """
    Resize an image to have its smaller dimension equal to target_size while maintaining aspect ratio.
    
    Args:
        img: PIL Image to resize
        target_size: Target size for the smaller dimension
        
    Returns:
        Resized PIL Image
    """
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img
