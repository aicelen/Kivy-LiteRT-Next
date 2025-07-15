from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from model import TensorFlowModel
import numpy as np


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
        """
        Example method
        """
        model = TensorFlowModel("my_model.tflite", use_gpu=True)
        data = np.random.randint(0, 100, size=(5, 5)) # use your data
        out = model.run_inference(data)
        print(out)


# Run the app
if __name__ == '__main__':
    MyApp().run()