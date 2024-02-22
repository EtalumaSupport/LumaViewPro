
def convert_12bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image // 16).astype(np.uint8)


def convert_12bit_to_16bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image * 16)
