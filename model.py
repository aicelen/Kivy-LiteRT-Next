from kivy.utils import platform

if platform =='android':
    from jnius import autoclass

    # Acquire Android context via PythonActivity
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
    context = PythonActivity.mActivity

    # Import Lite RT Next components (adjust package path as needed)
    CompiledModel = autoclass('com.google.ai.edge.litert.CompiledModel')
    Accelerator = autoclass('com.google.ai.edge.litert.Accelerator')
    Options = autoclass('com.google.ai.edge.litert.CompiledModel$Options')
    HashSet = autoclass('java.util.HashSet')


    class TensorFlowModel():
        """
        Class for inference of a .tflite model using LiteRT Next (version 2.0.1:alpha)
        Args:
            model_name(str):
        """
        def __init__(self, model_name: str, use_gpu: bool = True):
            
            if use_gpu:
                acc = Accelerator.GPU
            else:
                acc = Accelerator.CPU

            accelerator_set = HashSet()
            accelerator_set.add(acc)
            opts = Options(accelerator_set)

            self.model = CompiledModel.create(context.getAssets(), model_name, opts)

            self.input_buffers = self.model.createInputBuffers()
            self.output_buffers = self.model.createOutputBuffers()

        def run_inference(self, input_data):
            # Convert numpy array to flattened list
            if hasattr(input_data, 'tolist'):
                input_data = input_data.astype('float32').flatten().tolist()
            print(1)

            buf_in = self.input_buffers.get(0)
            buf_in.writeFloat(input_data)
            
            print(2)
            # Run inference
            self.model.run(self.input_buffers, self.output_buffers)
            print(3)
            # Read output from the first output tensor
            try:
                buf_out = self.output_buffers.get(0)
                result = buf_out.readFloat()
                
                # Depending on your output shape, you might need to reshape the result
                return result
            except:
                print('except')
                pass