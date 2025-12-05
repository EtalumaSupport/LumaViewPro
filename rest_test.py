import lumascope_api
import time

scope = lumascope_api.Lumascope()

# print(scope.get_current_position())
# scope.move_absolute_position('X',0,True)
# print(scope.get_current_position())
scope.capture()