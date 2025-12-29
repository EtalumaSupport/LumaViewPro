from fastapi import APIRouter, HTTPException, Depends
from rest_api.api_v1.api_config import get_scope, get_sequenced_capture_executor
from typing import TYPE_CHECKING
from lumascope_api import Lumascope

if TYPE_CHECKING:
    from ...modules.sequenced_capture_executor import SequencedCaptureExecutor

status_router = APIRouter(prefix="/status", tags=['Status'])

@status_router.get("", description="Returns status of motion and protocol")
def get_status(scope:Lumascope = Depends(get_scope),
               sequenced_capture_executor:"SequencedCaptureExecutor" = Depends(get_sequenced_capture_executor)):
    
    #Motion Status
    is_moving = scope.is_moving()
    motion_status = {'is_moving':is_moving}

    #Protocol Status
    protocol_running = sequenced_capture_executor.run_in_progress()
    if protocol_running:
        protocol_name = sequenced_capture_executor._sequence_name
        # scan_progress = str(sequenced_capture_executor.scan_count()) + '/' + str(sequenced_capture_executor.num_scans())
        current_scan = sequenced_capture_executor.scan_count()
        total_scans = sequenced_capture_executor.num_scans()
        # current_step = str(sequenced_capture_executor._curr_step + 1) + '/' + str(sequenced_capture_executor._protocol.num_steps())
        current_step = sequenced_capture_executor._curr_step + 1
        total_steps = sequenced_capture_executor._protocol.num_steps()
    else:
        protocol_name = None
        # scan_progress = None
        current_scan = None
        total_scans = None
        current_step = None
        total_steps = None
    
    protocol_status = {'protocol_running':protocol_running,
                       'protocol_name':protocol_name,
                    #    'scan_progress':scan_progress,
                       'current_scan':current_scan,
                       'total_scans':total_scans,
                       'current_step':current_step,
                       'total_steps':total_steps}
    
    return {'motion_status':motion_status,
            'protocol_status':protocol_status}