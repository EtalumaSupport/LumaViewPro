import requests
import unittest

#NOTE: Before running, please ensure that lumaviewpro.py is running locally and there is an existing protocol named 'test_protocol'

class TestRESTv1(unittest.TestCase):
    def test_position(self):
        response = requests.get('http://localhost:8000/api/v1/move/position')
        self.assertIsInstance(response.json(),dict)
        self.assertListEqual(list(response.json()),['X','Y','Z','T'])
        
    def test_move_absolute(self):
        body = {
            "motion_type": "absolute",
            "axis": "X",
            "um": 0,
            "overshoot_enabled": True,
            "ignore_limits": False
        }
        response = requests.post('http://localhost:8000/api/v1/move',json=body)
        self.assertDictEqual(response.json(),{"message": "Movement completed"})

    def test_move_relative(self):
        body = {
            "motion_type": "relative",
            "axis": "X",
            "um": 100,
            "overshoot_enabled": True,
            "ignore_limits": False
        }
        response = requests.post('http://localhost:8000/api/v1/move',json=body)
        self.assertDictEqual(response.json(),{"message": "Movement completed"})

    def test_move_validation(self):
        body = {
            "motion_type": "roolative",
            "axis": "x",
            "um": 0,
            "overshoot_enabled": True,
            "ignore_limits": False
        }
        response = requests.post('http://localhost:8000/api/v1/move',json=body)
        assert ('msg',"Input should be 'absolute' or 'relative'") in response.json()['detail'][0].items()
        assert ('msg',"Input should be 'X', 'Y', 'Z' or 'T'") in response.json()['detail'][1].items()

    def test_list_protocols(self):
        response = requests.get('http://localhost:8000/api/v1/protocol')
        self.assertIsInstance(response.json(),list)
        for item in response.json():
            assert '.tsv' in item

    def test_run_protocol(self):
        body = {
            "protocol_name": "test_protocol"
        }
        response = requests.post('http://localhost:8000/api/v1/protocol/run', json=body)
        self.assertDictEqual(response.json(),{"message": "Protocol started"})

    def test_live_capture(self):
        body = {
            "file_root": "img_",
            "append": "ms",
            "color": "BF",
            "tail_id_mode": "increment",
            "force_to_8bit": True,
            "output_format": "TIFF",
            "true_color": "BF",
            "timeout": 0,
            "all_ones_check": False,
            "sum_count": 1,
            "sum_delay_s": 0
        }
        response = requests.post('http://localhost:8000/api/v1/capture/live',json=body)
        self.assertDictEqual(response.json(),{"message": "Capture saved"})

    def test_status(self):
        response = requests.get('http://localhost:8000/api/v1/status')
        self.assertIsInstance(response.json(),dict)
        assert 'motion_status' in response.json() and 'protocol_status' in response.json()

if __name__ == '__main__':
    unittest.main()