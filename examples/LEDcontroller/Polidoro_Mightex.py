from mightex_device import MightexDevice
dev = MightexDevice() # Might automatically find device if one available
# if it is not found automatically, specify port directly
dev = MightexDevice(port='/dev/ttyUSB0') # Linux
dev = MightexDevice(port='/dev/tty.usbmodem262471') # Mac OS X
dev = MightexDevice(port='COM3') # Windows
dev.get_serial_number()
'04-150824-007'

dev.get_channel_count()
4

channel = 0 # channel numbering starts at 0, not 1!
dev.get_mode(channel)
'disable'

dev.set_normal_parameters(channel,1000,30)
dev.get_normal_parameters(channel)
{'current': 30, 'current_max': 1000}

dev.set_mode_normal(channel)
dev.get_load_voltage(channel)
2622

dev.set_normal_current(channel,200)
dev.get_load_voltage(channel)
3054

dev.set_mode_disable(channel)
dev.set_strobe_parameters(channel,100,1)
dev.get_strobe_parameters(channel)
{'current_max': 100, 'repeat': 1}

dev.set_strobe_profile_set_point(channel,0,100,1000000)
dev.set_strobe_profile_set_point(channel,1,10,500000)
dev.set_strobe_profile_set_point(channel,2,0,0)
profile = dev.get_strobe_profile(channel)

profile
[{'current': 100, 'duration': 1000000},
 {'current': 10, 'duration': 500000},
 {'current': 0, 'duration': 0}]

dev.set_mode_strobe(channel)
dev.get_strobe_profile(channel+1)

[{'current': 20, 'duration': 1000},
 {'current': 10, 'duration': 1000},
 {'current': 0, 'duration': 0}]

dev.set_strobe_profile(channel+1,profile)
dev.get_strobe_profile(channel+1)
dev.set_mode_strobe(channel+1)
dev.set_trigger_parameters(channel,100,True)
dev.get_trigger_parameters(channel)

{'current_max': 100, 'falling_edge': True}

dev.set_trigger_profile_set_point(channel,0,100,1000000)
dev.set_trigger_profile_set_point(channel,1,10,500000)
dev.set_trigger_profile_set_point(channel,2,0,0)
dev.get_trigger_profile(channel)

[{'current': 100, 'duration': 1000000},
 {'current': 10, 'duration': 500000},
 {'current': 0, 'duration': 0}]

dev.set_mode_trigger(channel)
dev.reset()
dev.get_trigger_profile(channel)

[{'current': 20, 'duration': 1000},
 {'current': 10, 'duration': 1000},
 {'current': 0, 'duration': 0}]

dev.set_trigger_profile_set_point(channel,0,100,1000000)
dev.set_trigger_profile_set_point(channel,1,10,500000)
dev.set_trigger_profile_set_point(channel,2,0,0)
dev.store_parameters()
dev.reset()
dev.get_trigger_profile(channel)

[{'current': 100, 'duration': 1000000},
 {'current': 10, 'duration': 500000},
 {'current': 0, 'duration': 0}]

dev.restore_factory_defaults()
dev.store_parameters()
dev.reset()
dev.get_trigger_profile(channel)

[{'current': 10, 'duration': 20},
 {'current': 0, 'duration': 0}]

from mightex_device import MightexDevices
devs = MightexDevices()  # Might automatically find all available devices
# if they are not found automatically, specify ports to use
devs = MightexDevices(use_ports=['/dev/ttyUSB0','/dev/ttyUSB1']) # Linux
devs = MightexDevices(use_ports=['/dev/tty.usbmodem262471','/dev/tty.usbmodem262472']) # Mac OS X
devs = MightexDevices(use_ports=['COM3','COM4']) # Windows
devs.keys()
dev = devs[serial_number]
