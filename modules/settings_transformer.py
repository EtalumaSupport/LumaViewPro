

class SettingsTransformer:

    def __init__(self):
        pass

    def _transform_intensity_settings(self, settings):

        scale_ratio = 255/100
      
        return {
            'min': {
                'min': round(settings['min']['min']*scale_ratio),
                'max': round(settings['min']['max']*scale_ratio),
            },
            'mean': {
                'min': round(settings['mean']['min']*scale_ratio),
                'max': round(settings['mean']['max']*scale_ratio),
            },
            'max': {
                'min': round(settings['max']['min']*scale_ratio),
                'max': round(settings['max']['max']*scale_ratio),
            }
        }

    def _transform_perimeter_settings(self, settings, pixels_per_um):
        return {
            'min': settings['min'] / pixels_per_um,
            'max': settings['max'] / pixels_per_um
        }

    def _transform_area_settings(self, settings, pixels_per_um):
        return {
            'min': settings['min'] / (pixels_per_um**2),
            'max': settings['max'] / (pixels_per_um**2)
        }

    def _transform_threshold_settings(self, setting):
        scale_ratio = 255/100
        return round(setting * scale_ratio)

    def _transform_context_settings(self, settings):
        return settings
        
    def _transform_segmentation_settings(self, settings):
        return {
            'algorithm': settings['algorithm'],
            'parameters': {
                'threshold': self._transform_threshold_settings(settings['parameters']['threshold'])
            }
        }

    def _transform_filter_settings(self, settings, pixels_per_um):
        return {
            'intensity': self._transform_intensity_settings(settings['intensity']),
            'perimeter': self._transform_perimeter_settings(settings['perimeter'], pixels_per_um),
            'area': self._transform_area_settings(settings['area'], pixels_per_um)
        }

    def transform_to_digital(self, settings):
        transformed_settings = {
            'context': self._transform_context_settings(settings['context']),
            'segmentation': self._transform_segmentation_settings(settings['segmentation']),
            'filters': self._transform_filter_settings(settings['filters'], pixels_per_um=settings['context']['pixels_per_um'])
        }
        return transformed_settings
