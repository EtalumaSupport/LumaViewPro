

import labware as lw


class CoordinateTransformer:

    def __init__(self):
        pass


    def stage_to_plate(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str,float],
        sx: float,
        sy: float
    ):
        # stage coordinates in um from bottom right
        # plate coordinates in mm from top left

        # Get labware dimensions
        dim_max = labware.get_dimensions()

        # Convert coordinates
        px = dim_max['x'] - (stage_offset['x'] + sx)/1000
        py = dim_max['y'] - (stage_offset['y'] + sy)/1000

        return px, py


    def plate_to_stage(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str,float],
        px: float,
        py: float
    ):
        # plate coordinates in mm from top left
        # stage coordinates in um from bottom right

        # Get labware dimensions
        dim_max = labware.get_dimensions()
 
        # Convert coordinates
        sx = dim_max['x'] - stage_offset['x']/1000 - px # in mm
        sy = dim_max['y'] - stage_offset['y']/1000 - py # in mm

        sx = sx*1000
        sy = sy*1000

        # return
        return sx, sy


    def plate_to_pixel(
        self,
        labware: lw.LabWare,
        px: float,
        py: float,
        scale_x: float,
        scale_y: float
    ):
        # plate coordinates in mm from top left
        # pixel coordinates in px from bottom left
        dim_max = labware.get_dimensions()

        # Convert coordinates
        pixel_x = px*scale_x
        pixel_y = (dim_max['y']-py)*scale_y

        return pixel_x, pixel_y


    def stage_to_pixel(
        self,
        labware: lw.LabWare,
        stage_offset: dict[str,float],
        sx: float,
        sy: float,
        scale_x: float,
        scale_y: float
    ):
        # stage coordinates in um from bottom right
        # plate coordinates in mm from top left
        # pixel coordinates in px from bottom left
        px, py = self.stage_to_plate(
            labware=labware,
            stage_offset=stage_offset,
            sx=sx,
            sy=sy
        )

        pixel_x, pixel_y = self.plate_to_pixel(
            labware=labware,
            px=px,
            py=py,
            scale_x=scale_x,
            scale_y=scale_y
        )

        return pixel_x, pixel_y
