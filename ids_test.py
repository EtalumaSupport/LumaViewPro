from camera.idscamera import IDSCamera

camera = IDSCamera()
print(camera.model_name)
print(camera._device_serial)
print(camera.get_pixel_format())
print(camera.get_supported_pixel_formats())
print(camera.get_exposure_t())