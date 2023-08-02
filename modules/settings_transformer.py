

class SettingsTransformer:

    def __init__(self):
        pass

    def _transform_intensity_settings(self, settings):

        scale_ratio = 255/100

        def _process_val(val):
            if val is None:
                return None
            
            return round(val*scale_ratio)


        return {
            'min': {
                'min': _process_val(settings['min']['min']),
                'max': _process_val(settings['min']['max']),
            },
            'mean': {
                'min': _process_val(settings['mean']['min']),
                'max': _process_val(settings['mean']['max']),
            },
            'max': {
                'min': _process_val(settings['max']['min']),
                'max': _process_val(settings['max']['max']),
            }
        }


    def _transform_perimeter_settings(self, settings, pixels_per_um):

        def _process_val(val):
            if val is None:
                return None
            
            return val / pixels_per_um

        return {
            'min': _process_val(settings['min']),
            'max': _process_val(settings['max'])
        }

    def _transform_sphericity_settings(self, settings):
        return settings


    def _transform_area_settings(self, settings, pixels_per_um):
        
        def _process_val(val):
            if val is None:
                return None
            
            return val / (pixels_per_um**2)

        return {
            'min': _process_val(settings['min']),
            'max': _process_val(settings['max'])
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
            'sphericity': self._transform_sphericity_settings(settings['sphericity']),
            'area': self._transform_area_settings(settings['area'], pixels_per_um)
        }

    def transform_to_digital(self, settings):
        transformed_settings = {
            'context': self._transform_context_settings(settings['context']),
            'segmentation': self._transform_segmentation_settings(settings['segmentation']),
            'filters': self._transform_filter_settings(settings['filters'], pixels_per_um=settings['context']['pixels_per_um'])
        }
        return transformed_settings
