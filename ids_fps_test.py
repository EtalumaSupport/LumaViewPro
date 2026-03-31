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

# Try setting throughput component BEFORE UserSetLoad
print("=== Before UserSetLoad ===")
try:
    comp = nm.FindNode('DeviceLinkThroughputLimitComponent')
    print(f'  Component = {comp.CurrentEntry().SymbolicValue()}')
    comp.SetCurrentEntry('Link')
    print(f'  Component -> Link (success!)')
except Exception as e:
    print(f'  Component set failed: {e}')

nm.FindNode('UserSetSelector').SetCurrentEntry('Default')
nm.FindNode('UserSetLoad').Execute()
nm.FindNode('UserSetLoad').WaitUntilDone()
print("\n=== After UserSetLoad ===")

# Check if UserSetLoad reset it
try:
    comp = nm.FindNode('DeviceLinkThroughputLimitComponent')
    print(f'  Component = {comp.CurrentEntry().SymbolicValue()}')
except: pass

# Try setting it again after UserSetLoad
try:
    comp = nm.FindNode('DeviceLinkThroughputLimitComponent')
    comp.SetCurrentEntry('Link')
    print(f'  Component -> Link (success!)')
except Exception as e:
    print(f'  Component set failed: {e}')

# Disable frame rate target
try:
    nm.FindNode('AcquisitionFrameRateTargetEnable').SetValue(False)
    print('  AcquisitionFrameRateTargetEnable -> False')
except Exception as e:
    print(f'  AcquisitionFrameRateTargetEnable: {e}')

# Max throughput
try:
    n = nm.FindNode('DeviceLinkThroughputLimit')
    n.SetValue(n.Maximum())
    print(f'  DeviceLinkThroughputLimit -> {n.Value()} B/s')
except Exception as e:
    print(f'  DeviceLinkThroughputLimit: {e}')

# Set resolution
nm.FindNode('Width').SetValue(1920)
nm.FindNode('Height').SetValue(1528)

# Now check frame rate limits
try:
    fr = nm.FindNode('AcquisitionFrameRate')
    print(f'\n  AcquisitionFrameRate = {fr.Value():.1f} (max={fr.Maximum():.1f})')
    fr.SetValue(fr.Maximum())
    print(f'  AcquisitionFrameRate -> {fr.Value():.1f}')
except Exception as e:
    print(f'  AcquisitionFrameRate: {e}')

try:
    limit = nm.FindNode('DeviceLinkAcquisitionFrameRateLimit')
    print(f'  DeviceLinkAcquisitionFrameRateLimit = {limit.Value():.1f}')
except: pass

nm.FindNode('ExposureTime').SetValue(10000)  # 10ms

ps = nm.FindNode('PayloadSize').Value()
print(f'\nResolution: {nm.FindNode("Width").Value()}x{nm.FindNode("Height").Value()}')
print(f'Payload size: {ps} bytes ({ps/1024/1024:.1f} MB)')

for i in range(6):
    buf = ds.AllocAndAnnounceBuffer(ps)
    ds.QueueBuffer(buf)

ds.StartAcquisition()
nm.FindNode('AcquisitionStart').Execute()

# Test 1: Raw grab
count = 0
start = time.monotonic()
for _ in range(100):
    buf = ds.WaitForFinishedBuffer(2000)
    ds.QueueBuffer(buf)
    count += 1
elapsed = time.monotonic() - start
print(f'\nRaw grab (no conversion): {count/elapsed:.1f} fps')

# Test 2: Full pipeline
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
