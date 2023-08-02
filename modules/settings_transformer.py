

class SettingsTransformer:

    def __init__(self, to_direction):
        self._to_direction = to_direction


    def _transform_intensity(self, settings, bitwidth=8):

        scale_ratio = (2**bitwidth-1)/100

        def _process_val(val):
            if val is None:
                return None
            
            if self._to_direction=='digital':
                return round(val*scale_ratio)
            else:
                return round(val/scale_ratio)


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


    def _transform_perimeter(self, settings, pixels_per_um):

        def _process_val(val):
            if val is None:
                return None
            
            if self._to_direction=='digital':
                return val * pixels_per_um
            else:
                return val / pixels_per_um

        return {
            'min': _process_val(settings['min']),
            'max': _process_val(settings['max'])
        }

    def _transform_sphericity(self, settings):
        return settings


    def _transform_area(self, settings, pixels_per_um):
        
        def _process_val(val):
            if val is None:
                return None
            
            if self._to_direction=='digital':
                return val * (pixels_per_um**2)
            else:
                return val / (pixels_per_um**2)

        return {
            'min': _process_val(settings['min']),
            'max': _process_val(settings['max'])
        }


    def _transform_threshold(self, settings, bitwidth=8):
        scale_ratio = (2**bitwidth-1)/100

        if self._to_direction=='digital':
            return round(settings * scale_ratio)
        else:
            return round(settings / scale_ratio)


    def _transform_context(self, settings):
        return settings
        
    def _transform_segmentation(self, settings):
        return {
            'algorithm': settings['algorithm'],
            'parameters': {
                'threshold': self._transform_threshold(settings=settings['parameters']['threshold'])
            }
        }

    def _transform_filters(self, settings, pixels_per_um):
        return {
            'intensity': self._transform_intensity(settings=settings['intensity']),
            'perimeter': self._transform_perimeter(settings=settings['perimeter'], pixels_per_um=pixels_per_um),
            'sphericity': self._transform_sphericity(settings=settings['sphericity']),
            'area': self._transform_area(settings=settings['area'], pixels_per_um=pixels_per_um)
        }
        
    def transform(self, settings):
        return {
            'context': self._transform_context(settings=settings['context']),
            'segmentation': self._transform_segmentation(settings=settings['segmentation']),
            'filters': self._transform_filters(settings=settings['filters'], pixels_per_um=settings['context']['pixels_per_um'])
        }
