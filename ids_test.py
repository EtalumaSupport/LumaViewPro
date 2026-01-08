from camera.idscamera import IDSCamera

camera = IDSCamera()

print([value for value in camera.remote_nodemap.FindNode("BinningHorizontal").ValidValues()])

camera.disconnect()