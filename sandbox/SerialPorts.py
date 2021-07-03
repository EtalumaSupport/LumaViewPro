
# List all ports and properties
import serial.tools.list_ports
ports = serial.tools.list_ports.comports(include_links = True)

for port in ports:
    print('-------------------------------------------')
    print('device:       ', port.device)
    print('name:         ', port.name)
    print('description:  ', port.description)
    print('hwid:         ', port.hwid)
    print('vid:          ', port.vid) # vendor ID
    print('pid:          ', port.pid) # product ID
    print('serial_number:', port.serial_number)
    print('location:     ', port.location)
    print('manufacturer: ', port.manufacturer)
    print('product:      ', port.product)
    print('interface:    ', port.interface)
    print('-------------------------------------------')
    if (port.vid == 1155) and (port.pid == 22336):
        print('LED Control Board identified')
    if (port.vid == 10812) and (port.pid == 256):
        print('Trinamic Motor Control Board identified')
    print('')
