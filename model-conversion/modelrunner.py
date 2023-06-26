## onnxruntime inference session wrapper that loads custom libs
import os
import sys
import numpy as np
import onnxruntime

PREFER_CUSTOMOP_DEBUGLIB = True # in case both debug and release custom ops exist, debug is chosen

class ModelRunner(object):
    def __init__(self, modelPath, sess_options=None, providers=[ "CUDAExecutionProvider"], register_custom_ops=True):
        if sess_options:
            self.sess_opts = onnxruntime.SessionOptions(sess_options)
        else:
            self.sess_opts = onnxruntime.SessionOptions()
        
        if register_custom_ops:
            shared_library = self.get_nmp_libpath()
            onnxruntime.set_default_logger_severity(3) # disable warnings about multiple op registration
            self.sess_opts.register_custom_ops_library(shared_library)
            print("Registered custom ops library: " + shared_library)
        
        self.sess = onnxruntime.InferenceSession(modelPath, sess_options=self.sess_opts, providers=providers)
        # if providers:
        #     self.sess.set_providers(providers)

        onnxruntime.set_default_logger_severity(2) #reenable warnings

    def get_input_node_name(self, index=0):
        return self.sess.get_inputs()[index].name
    
    def get_num_inputs(self):
        return len(self.sess.get_inputs())

    # inputDict is a dictionary containing the model inputs: keys are model input names, values are expected to be numpy arrays.
    def run(self, inputDict, runOptions=None):
        pred_onnx = self.sess.run(None, inputDict, run_options=self.wrap_run_options(runOptions))
        return pred_onnx
    
    def wrap_run_options(self, sessionOptionDict):
        # Create an onnxruntime.RunOptions object from the passed parameter dict.
        return None

    def get_nmp_libpath(self):
        filepath = os.path.dirname(os.path.realpath(__file__))
        libname = "-CustomOps"
        libpath = os.path.join(filepath, "..", "build", "src", "ort_custom_ops")

        if sys.platform.startswith("win"):
            shared_library_debug = os.path.join(libpath, "Debug", libname + 'd.dll')
            shared_library = os.path.join(libpath, "Release", libname + '.dll')
            if not os.path.exists(shared_library) and not os.path.exists(shared_library_debug)  :
                raise FileNotFoundError("Unable to find '{0} or {1}'".format(shared_library, shared_library_debug))
            elif not os.path.exists(shared_library) or (PREFER_CUSTOMOP_DEBUGLIB and os.path.exists(shared_library_debug)):
                return shared_library_debug
        elif sys.platform.startswith("darwin"):
            shared_library = os.path.join(libpath, 'lib' + libname + '.dylib')
            shared_librarydebug = os.path.join(libpath, 'lib' + libname + "d" + '.dylib')
            if not os.path.exists(shared_library) and not os.path.exists(shared_librarydebug)  :
                raise FileNotFoundError("Unable to find '{0} or {1}'".format(shared_library, shared_librarydebug))
            elif not os.path.exists(shared_library) or (PREFER_CUSTOMOP_DEBUGLIB and os.path.exists(shared_librarydebug)):
                return shared_librarydebug
        else:
            shared_library = os.path.join(libpath,'lib' + libname + '.so')
            #print("shared custom", shared_library)
            shared_librarydebug = os.path.join(libpath, 'lib' + libname + "d" + '.so')
            if not os.path.exists(shared_library) and not os.path.exists(shared_librarydebug)  :
                raise FileNotFoundError("Unable to find '{0} or {1}'".format(shared_library, shared_librarydebug))
            elif not os.path.exists(shared_library) or (PREFER_CUSTOMOP_DEBUGLIB and os.path.exists(shared_librarydebug)):
                return shared_librarydebug
        
        return shared_library