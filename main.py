from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from model import TensorFlowModel
from PIL import Image
import numpy as np
from time import time

class MyApp(App):
    def build(self):
        # Create a layout to hold the button
        layout = BoxLayout(orientation='vertical')
        
        # Create a button
        button = Button(
            text='Click Me!',
            size_hint=(0.5, 0.5),  # Button will be half the size of the window
            pos_hint={'center_x': 0.5, 'center_y': 0.5}  # Center the button
        )
        
        # Bind the button press event to a callback function
        button.bind(on_press=self.test_model)
        
        # Add the button to the layout
        layout.add_widget(button)
        
        return layout
    
    def test_model(self, instance):
        print("Button was pressed!")
        model = "model_working.tflite"

        if model == "model_working.tflite":
            model = TensorFlowModel(model)

            image = Image.open('test.png')
            image = image.convert('L') 
            image = image.resize((40, 70), Image.NEAREST)
            input_data = np.array(image).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)  # [height, width, channels] -> [1, height, width, channels]
            input_data = np.expand_dims(input_data, axis=-1)  # [1, height, width] -> [1, height, width, 1]
            a = time()
            out = model.run_inference(input_data)
            print(time()-a)
            print(out)

        else:
            model = TensorFlowModel(model)

            image = Image.open('test.png')
            image = image.convert('RGB')
            input_data = np.array(image).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)  # [height, width, channels] -> [1, height, width, channels]
            print(input_data.shape)
            print('starting')
            a = time()
            out = model.run_inference(input_data)
            print(time()-a)

        

# Run the app
if __name__ == '__main__':
    MyApp().run()