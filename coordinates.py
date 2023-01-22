class coordinate_system:
    """ A class for tranforming coordinate systems in the microscope:
    Base coordinate system is assumed to be in um
    with an origin and positive orientation of the motor controls
    i.e. positive um motion is aligned with increase in 'steps' """

    def __init__(self):
        """ Initialize um coordinates to a homed x, y, z stage and a known working area"""
        self.x = 0        # x coordinate in um
        self.y = 0        # y coordinate in um
        self.z = 0        # z coordinate in um
        self.w = 120000   # width in x-direction
        self.h = 80000    # height in y-direction
        self.v = 14000    # vertical in z-direction
        self.offset_x = 0 # from plate to stage offset from top in x-dir
        self.offset_y = 0 # from plate to stage offset from top in x-dir
        self.scale_x = 1  # scale from um to pixels in x_direction
        self.scale_y = 1  # scale from um to pixels in x_direction

    def to_stage(self, x, y): # from above
        """ Return coordinate position of the stage viewing from front-above
        in microns with the lower left as the origin,
        x-pos positive to the right, y-pos positive to the rear"""

        stage_x = self.w-self.x
        stage_y = self.y

        return stage_x, stage_y

    def to_plate(self, x, y): # from above
        """ Return coordinate position of a plate viewing from front-above
        in microns with the lower left as the origin,
        x-pos positive to the right, y-pos positive to the rear"""

        stage_x, stage_y = to_stage(x, y)
        plate_x = stage_x + self.offset_x
        plate_y = stage_y + self.offset_y

        return plate_x, plate_y
    
    def to_pixels(self, x, y): # from above
        """ Return coordinate position of a plate viewing from front-above
        in pixels with the lower left as the origin,
        x-pos positive to the right, y-pos positive to the rear"""

        plate_x, plate_y = to_plate(x, y)
        pixel_x = plate_x*self.scale_x
        pixel_y = plate_y*self.scale_y

        return pixel_x, pixel_y
