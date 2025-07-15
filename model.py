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

    # buildozer is sometimes cutting unused classes away
    # but Tensorbuffer is used run_inference()
    # importing stops buildozer from cutting it away
    TensorBuffer = autoclass('com.google.ai.edge.litert.Tensorbuffer')


    class TensorFlowModel():
        """
        Class for inference of a .tflite model using LiteRT Next (version 2.0.1:alpha)
        Args:
            model_name(str): Name of the model for inference (needs to be located in assets folder)
            use_gpu(bool): If True uses GPU acceleration
        """
        def __init__(self, model_name: str, use_gpu: bool = False):
            if use_gpu:
                acc = Accelerator.GPU
            else:
                acc = Accelerator.CPU

            # create accelerator
            accelerator_set = HashSet()
            accelerator_set.add(acc)
            opts = Options(accelerator_set)

            # create model
            self.model = CompiledModel.create(context.getAssets(), model_name, opts)

            # create input and output buffers
            self.input_buffers = self.model.createInputBuffers()
            self.output_buffers = self.model.createOutputBuffers()

        def run_inference(self, input_data):
            """
            Method of TensorFlowModel class performing inference of chosen model
            Args:
                input_data(np.array): Input data for the model
            Returns:
                result: Ouput of model 
            """
            # Convert numpy array to flattened list
            if hasattr(input_data, 'tolist'):
                input_data = input_data.astype('float32').flatten().tolist()

            # fill buffer with input data
            buf_in = self.input_buffers.get(0)
            buf_in.writeFloat(input_data)
            
            # run inference
            self.model.run(self.input_buffers, self.output_buffers)

            # read output from the first output tensor
            buf_out = self.output_buffers.get(0)
            result = buf_out.readFloat()
            
            return result