"""IDS camera raw FPS test — bypasses LVP display pipeline."""
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl
import time

ids_peak.Library.Initialize()
dm = ids_peak.DeviceManager.Instance()
dm.Update()
d = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
ds = d.DataStreams()[0].OpenDataStream()
nm = d.RemoteDevice().NodeMaps()[0]

nm.FindNode('UserSetSelector').SetCurrentEntry('Default')
nm.FindNode('UserSetLoad').Execute()
nm.FindNode('UserSetLoad').WaitUntilDone()

# Disable frame rate limiter
try:
    nm.FindNode('AcquisitionFrameRateTargetEnable').SetValue(False)
    print('Frame rate limiter disabled')
except Exception as e:
    print(f'Frame rate limiter: {e}')

# Max throughput
try:
    n = nm.FindNode('DeviceLinkThroughputLimit')
    n.SetValue(n.Maximum())
    print(f'Throughput limit: {n.Value()} B/s')
except Exception as e:
    print(f'Throughput: {e}')

nm.FindNode('ExposureTime').SetValue(10000)  # 10ms
nm.FindNode('Width').SetValue(1920)
nm.FindNode('Height').SetValue(1528)
w = nm.FindNode('Width').Value()
h = nm.FindNode('Height').Value()
print(f'Exposure: 10ms, Resolution: {w}x{h}')

ps = nm.FindNode('PayloadSize').Value()
print(f'Payload size: {ps} bytes ({ps/1024/1024:.1f} MB)')

for i in range(6):
    buf = ds.AllocAndAnnounceBuffer(ps)
    ds.QueueBuffer(buf)

ds.StartAcquisition()
nm.FindNode('AcquisitionStart').Execute()

# Test 1: Raw grab loop - no conversion
count = 0
start = time.monotonic()
for _ in range(100):
    buf = ds.WaitForFinishedBuffer(2000)
    ds.QueueBuffer(buf)
    count += 1
elapsed = time.monotonic() - start
print(f'\nRaw grab (no conversion): {count/elapsed:.1f} fps')

# Test 2: With BufferToImage + early return
count = 0
start = time.monotonic()
for _ in range(100):
    buf = ds.WaitForFinishedBuffer(2000)
    img = ids_peak_ipl_extension.BufferToImage(buf)
    ds.QueueBuffer(buf)
    count += 1
elapsed = time.monotonic() - start
print(f'BufferToImage + early return: {count/elapsed:.1f} fps')

# Test 3: Full pipeline (BufferToImage + ConvertTo + numpy copy)
count = 0
start = time.monotonic()
for _ in range(100):
    buf = ds.WaitForFinishedBuffer(2000)
    img = ids_peak_ipl_extension.BufferToImage(buf)
    ds.QueueBuffer(buf)
    img = img.ConvertTo(ids_peak_ipl.PixelFormatName_Mono8)
    frame = img.get_numpy().copy()
    count += 1
elapsed = time.monotonic() - start
print(f'Full pipeline (convert+copy): {count/elapsed:.1f} fps')

nm.FindNode('AcquisitionStop').Execute()
ds.StopAcquisition()
ids_peak.Library.Close()
print('\nDone')
