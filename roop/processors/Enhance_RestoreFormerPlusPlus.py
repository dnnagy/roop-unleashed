from typing import Any, List, Callable
import cv2 
import threading
import numpy as np
import onnxruntime
import onnx
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


# THREAD_LOCK = threading.Lock()


class Enhance_RestoreFormerPlusPlus():
    model_restoreformerplusplus = None
    name = None # Idk why, but needed, see https://github.com/C0untFloyd/roop-unleashed/issues/350#issuecomment-1826484047
    devicename = None

    processorname = 'restoreformerplusplus'
    type = 'enhance'
    

    def Initialize(self, devicename:str):
        if self.model_restoreformerplusplus is None:
            # replace Mac mps with cpu for the moment
            devicename = devicename.replace('mps', 'cpu')
            self.devicename = devicename
            model_path = resolve_relative_path('../models/RestoreFormerPlusPlus/RestoreFormerPlusPlus.onnx')
            self.model_restoreformerplusplus = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_restoreformerplusplus.get_inputs()
            model_outputs = self.model_restoreformerplusplus.get_outputs()
            self.io_binding = self.model_restoreformerplusplus.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)
            print(f"RestoreFormerPlusPlus model initialized with the following props:")
            for j,inp in enumerate(self.model_inputs):
                print(f"inputs[{j}]:", inp)
            for j,oup in enumerate(model_outputs):
                print(f"outputs[{j}]:", oup)

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)
        
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame) # .astype(np.float32)
        self.model_restoreformerplusplus.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs 
        
        # post-process
        result = np.clip(result, -1.0, 1.0)
        result = (result + 1.0) / 2.0
        result = result * 255.0
        result = result.transpose(1, 2, 0)
        #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)    
        return result.astype(np.uint8), scale_factor


    def Release(self):
        del self.model_restoreformerplusplus
        self.model_restoreformerplusplus = None
        del self.io_binding
        self.io_binding = None

